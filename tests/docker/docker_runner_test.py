"""Runner-container roundtrips: packager → emulator → docker → result.

Each test packages a real function with kinetic's own packager, seeds
the artifacts into the emulator, executes the real runner image with
the command line derived from the actual Job spec, and asserts on the
result payload read back from the emulator — the full remote-execution
contract with no Kubernetes and no mocks.
"""

import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import uuid

import cloudpickle
from absl.testing import absltest

from kinetic.data.data import make_data_ref
from kinetic.utils import packager
from tests.docker.docker_fixture import (
  DockerTierTestCase,
  derive_docker_command,
)

_RUN_TIMEOUT_SECONDS = 300


def _sha256_file(path):
  with open(path, "rb") as f:
    return hashlib.sha256(f.read()).hexdigest()


# ----------------------------------------------------------------------
# Payload entry points. Module level so the packager ships them by value
# (package_root covers this directory); each imports what it needs
# inside the body because it executes in the container.
# ----------------------------------------------------------------------


def _add(a, b):
  return a + b


def _boom():
  raise ValueError("boom from the container")


def _read_env(name):
  import os

  return os.environ.get(name)


def _job_env_snapshot():
  import os

  return {
    key: os.environ.get(key)
    for key in ("KERAS_BACKEND", "JAX_PLATFORMS", "JOB_ID", "GCS_BUCKET")
  }


def _list_dir(path):
  import os

  return sorted(os.listdir(path))


def _read_file(path):
  """Report what a single-file Data ref resolved to, and its bytes."""
  import os

  return {
    "name": os.path.basename(path),
    "is_file": os.path.isfile(path),
    "content": open(path).read(),
  }


def _probe_mount(path):
  """List *path* and report whether it rejects writes (GKE FUSE is ro)."""
  import os

  listing = sorted(os.listdir(path))
  try:
    with open(os.path.join(path, "probe.tmp"), "w") as f:
      f.write("x")
    writable = True
  except OSError:
    writable = False
  return listing, writable


def _import_shipped_and_read():
  import kinetic_shipped_mod

  with open("data.txt") as f:
    return kinetic_shipped_mod.VALUE, f.read()


def _six_version():
  import six

  return six.__version__


@dataclasses.dataclass
class _SubmitOutcome:
  proc: subprocess.CompletedProcess
  result: dict
  job_id: str
  bucket: str


class TestDockerRunnerRoundtrip(DockerTierTestCase):
  def _submit(
    self,
    func,
    args=(),
    kwargs=None,
    env_vars=None,
    zip_entries=None,
    plan_json=None,
    corrupt_payload_sha=False,
    requirements=None,
    fuse_volume_specs=None,
    fuse_host_dirs=None,
  ):
    """Package *func*, seed the emulator, run the container, collect."""
    bucket = self.server.make_bucket(suffix="docker")
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    tmp_path = pathlib.Path(tmp.name)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name, content in (zip_entries or {"dummy.py": "x = 1"}).items():
      entry = workspace / name
      entry.parent.mkdir(parents=True, exist_ok=True)
      entry.write_text(content)
    context_path = str(tmp_path / "context.zip")
    packager.zip_working_dir(str(workspace), context_path, plan_json=plan_json)

    payload_path = str(tmp_path / "payload.pkl")
    packager.save_payload(
      func,
      args,
      kwargs or {},
      env_vars or {},
      payload_path,
      package_root=os.path.dirname(__file__),
    )

    self.server.write_blob(
      bucket, f"{job_id}/context.zip", pathlib.Path(context_path).read_bytes()
    )
    self.server.write_blob(
      bucket, f"{job_id}/payload.pkl", pathlib.Path(payload_path).read_bytes()
    )
    requirements_uri = None
    if requirements is not None:
      self.server.write_blob(bucket, f"{job_id}/requirements.txt", requirements)
      requirements_uri = f"gs://{bucket}/{job_id}/requirements.txt"

    command = derive_docker_command(
      self.image,
      bucket,
      job_id,
      self.server.port,
      requirements_uri=requirements_uri,
      payload_sha256=(
        "0" * 64 if corrupt_payload_sha else _sha256_file(payload_path)
      ),
      context_sha256=_sha256_file(context_path),
      fuse_volume_specs=fuse_volume_specs,
      fuse_host_dirs=fuse_host_dirs,
    )
    # A timed-out subprocess.run only kills the docker CLI, not the
    # container, and --rm never fires for a running container — force
    # removal so a hang cannot leak it.
    self.addCleanup(
      subprocess.run,
      ["docker", "rm", "-f", f"kinetic-test-{job_id}"],
      capture_output=True,
    )
    proc = subprocess.run(
      command, capture_output=True, text=True, timeout=_RUN_TIMEOUT_SECONDS
    )

    result_bytes = self.server.read_blob(bucket, f"{job_id}/result.pkl")
    if result_bytes is None:
      # Every runner phase — even pre-execution failures — uploads a
      # result payload; its absence means the container died before the
      # runner could report. Surface the container's own output instead
      # of letting assertions crash on a None result.
      raise RuntimeError(
        f"Container exited without writing result.pkl.\n"
        f"Exit code: {proc.returncode}\n"
        f"Stderr:\n{proc.stderr}\n"
        f"Stdout:\n{proc.stdout}"
      )
    return _SubmitOutcome(
      proc=proc,
      result=cloudpickle.loads(result_bytes),
      job_id=job_id,
      bucket=bucket,
    )

  # ------------------------------------------------------------------

  def test_image_has_no_entrypoint(self):
    """derive_docker_command's command mapping relies on this."""
    inspect = subprocess.run(
      [
        "docker",
        "image",
        "inspect",
        "--format",
        "{{json .Config.Entrypoint}}",
        self.image,
      ],
      capture_output=True,
      text=True,
    )
    self.assertEqual(inspect.returncode, 0, inspect.stderr)
    self.assertIsNone(json.loads(inspect.stdout))

  def test_success_roundtrip(self):
    outcome = self._submit(_add, args=(2, 3))

    self.assertEqual(outcome.proc.returncode, 0, outcome.proc.stderr)
    self.assertTrue(outcome.result["success"])
    self.assertEqual(outcome.result["result"], 5)
    self.assertIsNone(outcome.result["exception"])
    self.assertEqual(outcome.result["phase"], "execute")

  def test_user_exception_comes_back_with_traceback(self):
    outcome = self._submit(_boom)

    self.assertEqual(outcome.proc.returncode, 1, outcome.proc.stderr)
    self.assertFalse(outcome.result["success"])
    self.assertIn("boom from the container", str(outcome.result["exception"]))
    self.assertIn("ValueError", outcome.result["traceback"])
    self.assertEqual(outcome.result["phase"], "execute")

  def test_payload_sha_mismatch_is_rejected_before_unpickling(self):
    outcome = self._submit(_add, args=(2, 3), corrupt_payload_sha=True)

    self.assertEqual(outcome.proc.returncode, 1, outcome.proc.stderr)
    self.assertFalse(outcome.result["success"])
    self.assertEqual(outcome.result["phase"], "artifact verification")
    self.assertIn(
      "Security verification failed", str(outcome.result["exception"])
    )

  def test_job_spec_env_reaches_the_container(self):
    outcome = self._submit(_job_env_snapshot)

    self.assertEqual(outcome.proc.returncode, 0, outcome.proc.stderr)
    snapshot = outcome.result["result"]
    self.assertEqual(snapshot["KERAS_BACKEND"], "jax")
    self.assertEqual(snapshot["JAX_PLATFORMS"], "cpu")
    self.assertEqual(snapshot["JOB_ID"], outcome.job_id)
    self.assertEqual(snapshot["GCS_BUCKET"], outcome.bucket)

  def test_user_env_vars_applied_before_execution(self):
    outcome = self._submit(
      _read_env,
      args=("KINETIC_DOCKER_TIER_VAR",),
      env_vars={"KINETIC_DOCKER_TIER_VAR": "hello-from-host"},
    )

    self.assertEqual(outcome.proc.returncode, 0, outcome.proc.stderr)
    self.assertEqual(outcome.result["result"], "hello-from-host")

  def test_workspace_plan_drives_imports_and_cwd(self):
    outcome = self._submit(
      _import_shipped_and_read,
      zip_entries={
        "src/kinetic_shipped_mod.py": "VALUE = 7",
        "run/data.txt": "hello",
      },
      plan_json={"sys_path_rel": ["", "src"], "client_cwd_rel": "run"},
    )

    self.assertEqual(outcome.proc.returncode, 0, outcome.proc.stderr)
    self.assertEqual(tuple(outcome.result["result"]), (7, "hello"))

  def test_data_ref_downloads_from_the_emulator(self):
    data_bucket = self.server.make_bucket(suffix="data")
    self.server.write_blob(data_bucket, "cache/h1/part.txt", b"payload")
    ref = {
      "__data_ref__": True,
      "uri": f"gs://{data_bucket}/cache/h1",
      "is_dir": True,
      "mount_path": None,
    }

    outcome = self._submit(_list_dir, args=(ref,))

    self.assertEqual(outcome.proc.returncode, 0, outcome.proc.stderr)
    self.assertEqual(outcome.result["result"], ["part.txt"])

  def test_single_object_data_ref_downloads_as_a_file(self):
    """``Data("gs://bucket/dir/file.h5")`` reaches the pod as that file."""
    data_bucket = self.server.make_bucket(suffix="data")
    for name, content in (
      ("datasets/weights.h5", b"the weights"),
      ("datasets/notes.txt", b"unrelated"),
      ("datasets/other.h5", b"also unrelated"),
    ):
      self.server.write_blob(data_bucket, name, content)
    ref = make_data_ref(f"gs://{data_bucket}/datasets/weights.h5", False)

    outcome = self._submit(_read_file, args=(ref,))

    self.assertEqual(outcome.proc.returncode, 0, outcome.proc.stderr)
    self.assertEqual(
      outcome.result["result"],
      {"name": "weights.h5", "is_file": True, "content": "the weights"},
    )

  def test_fuse_single_object_resolves_out_of_its_mounted_parent(self):
    """GCS FUSE mounts the parent; the runner picks the object by name."""
    host_dir = tempfile.TemporaryDirectory()
    self.addCleanup(host_dir.cleanup)
    for name, content in (
      ("weights.h5", b"the weights"),
      ("notes.txt", b"unrelated"),
      ("other.h5", b"also unrelated"),
    ):
      pathlib.Path(host_dir.name, name).write_bytes(content)
    mount_path = "/mnt/kinetic-data/0"
    gcs_uri = "gs://some-bucket/datasets/weights.h5"
    ref = make_data_ref(gcs_uri, False, mount_path=mount_path, fuse=True)
    fuse_spec = {
      "gcs_uri": gcs_uri,
      "mount_path": mount_path,
      "is_dir": False,
      "read_only": True,
    }

    outcome = self._submit(
      _read_file,
      args=(ref,),
      fuse_volume_specs=[fuse_spec],
      fuse_host_dirs={mount_path: host_dir.name},
    )

    self.assertEqual(outcome.proc.returncode, 0, outcome.proc.stderr)
    self.assertEqual(
      outcome.result["result"],
      {"name": "weights.h5", "is_file": True, "content": "the weights"},
    )

  def test_fuse_mount_from_the_job_spec_is_visible_and_read_only(self):
    """The mount path and ro flag come from the spec's volumeMounts —
    built by the real build_gcs_fuse_v1_volumes machinery — with a host
    directory standing in for the bucket, mirroring GKE's readOnly."""
    host_dir = tempfile.TemporaryDirectory()
    self.addCleanup(host_dir.cleanup)
    pathlib.Path(host_dir.name, "shard-0.tfrecord").write_bytes(b"x")
    mount_path = "/mnt/kinetic-data/0"
    gcs_uri = "gs://some-bucket/data/train"
    ref = make_data_ref(gcs_uri, True, mount_path=mount_path, fuse=True)
    fuse_spec = {
      "gcs_uri": gcs_uri,
      "mount_path": mount_path,
      "is_dir": True,
      "read_only": True,
    }

    outcome = self._submit(
      _probe_mount,
      args=(ref,),
      fuse_volume_specs=[fuse_spec],
      fuse_host_dirs={mount_path: host_dir.name},
    )

    self.assertEqual(outcome.proc.returncode, 0, outcome.proc.stderr)
    listing, writable = outcome.result["result"]
    self.assertEqual(listing, ["shard-0.tfrecord"])
    self.assertFalse(writable, "GKE FUSE mounts are readOnly; ro must hold")

  def test_requirements_uri_triggers_a_real_uv_install(self):
    # six 1.17.0 is already baked into the image (a `kubernetes`
    # transitive dep), so pinning the current version would pass even
    # if the install never ran. Pinning a DOWNGRADE proves uv actually
    # mutated the environment.
    baseline = self._submit(_six_version)
    self.assertEqual(baseline.proc.returncode, 0, baseline.proc.stderr)
    self.assertEqual(baseline.result["result"], "1.17.0")

    outcome = self._submit(_six_version, requirements="six==1.16.0\n")

    self.assertEqual(outcome.proc.returncode, 0, outcome.proc.stderr)
    self.assertEqual(outcome.result["result"], "1.16.0")


if __name__ == "__main__":
  absltest.main()
