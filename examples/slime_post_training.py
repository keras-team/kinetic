import kinetic
from kinetic import Data

SLIME_BASE_REPO = "us-docker.pkg.dev/your-project-id/kinetic-slime"


@kinetic.submit(
  accelerator="gpu-h100x8",
  container_image="prebuilt",
  base_image_repo=SLIME_BASE_REPO,
  capture_env_vars=["HF_TOKEN", "WANDB_*"],
)
def run_slime_glm4_quickstart(
  checkpoint_dir: str,
  num_rollout: int = 2,
):
  import subprocess
  from pathlib import Path

  local_save = Path(checkpoint_dir)
  local_ref = Path("/tmp/slime-output/GLM-Z1-9B-0414_torch_dist")
  local_save.mkdir(parents=True, exist_ok=True)

  setup = r"""
set -euxo pipefail
cd /root/slime

huggingface-cli download zai-org/GLM-Z1-9B-0414 \
  --local-dir /root/GLM-Z1-9B-0414
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024

source scripts/models/glm4-9B.sh
if [ ! -d /tmp/slime-output/GLM-Z1-9B-0414_torch_dist ]; then
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/GLM-Z1-9B-0414 \
    --save /tmp/slime-output/GLM-Z1-9B-0414_torch_dist
fi
"""
  subprocess.run(["bash", "-lc", setup], check=True)

  script_path = Path("/root/slime/scripts/run-glm4-9B.sh")
  run_script = script_path.read_text()
  run_script = run_script.replace(
    "--ref-load /root/GLM-Z1-9B-0414_torch_dist",
    f"--ref-load {local_ref}",
  )
  run_script = run_script.replace(
    "--load /root/GLM-Z1-9B-0414_slime/",
    f"--load {local_save}/",
  )
  run_script = run_script.replace(
    "--save /root/GLM-Z1-9B-0414_slime/",
    f"--save {local_save}/",
  )
  run_script = run_script.replace(
    "--num-rollout 3000",
    f"--num-rollout {num_rollout}",
  )

  patched = Path("/tmp/run-glm4-9B-kinetic.sh")
  patched.write_text(run_script)
  patched.chmod(0o755)

  subprocess.run(["bash", str(patched)], check=True)

  return {"checkpoints": str(local_save)}


if __name__ == "__main__":
  job = run_slime_glm4_quickstart(
    Data("gs://your-bucket/slime-runs/GLM-Z1-9B-0414_slime/")
  )
  print(f"Submitted Kinetic job: {job.job_id}")
  print(job.result())
