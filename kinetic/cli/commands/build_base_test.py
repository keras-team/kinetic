"""Tests for kinetic.cli.commands.build_base — build prebuilt base images."""

from unittest import mock
from unittest.mock import MagicMock

from absl.testing import absltest
from click.testing import CliRunner
from google.api_core import exceptions as google_exceptions

from kinetic.cli.commands.build_base import (
  _ensure_dockerhub_secrets,
  _is_ar_repo,
  _parse_ar_repo,
  _secret_exists,
  build_base,
)

_MODULE = "kinetic.cli.commands.build_base"


class TestSecretExists(absltest.TestCase):
  def test_returns_true_when_found(self):
    sm_client = MagicMock()
    self.assertTrue(_secret_exists(sm_client, "proj", "my-secret"))
    sm_client.get_secret.assert_called_once()

  def test_returns_false_when_not_found(self):
    sm_client = MagicMock()
    sm_client.get_secret.side_effect = google_exceptions.NotFound("nope")
    self.assertFalse(_secret_exists(sm_client, "proj", "my-secret"))

  def test_passes_correct_resource_name(self):
    sm_client = MagicMock()
    _secret_exists(sm_client, "my-proj", "dockerhub-token")
    request = sm_client.get_secret.call_args.kwargs["request"]
    self.assertEqual(
      request["name"],
      "projects/my-proj/secrets/dockerhub-token",
    )


class TestEnsureDockerHubSecrets(absltest.TestCase):
  def test_skips_prompt_when_secrets_exist(self):
    with (
      mock.patch(
        f"{_MODULE}.secretmanager.SecretManagerServiceClient",
      ) as mock_cls,
      mock.patch(f"{_MODULE}._secret_exists", return_value=True),
      mock.patch(f"{_MODULE}.click.prompt") as mock_prompt,
    ):
      _ensure_dockerhub_secrets("proj")

    mock_prompt.assert_not_called()
    mock_cls.return_value.create_secret.assert_not_called()

  def test_prompts_and_creates_when_missing(self):
    mock_sm = MagicMock()
    with (
      mock.patch(
        f"{_MODULE}.secretmanager.SecretManagerServiceClient",
        return_value=mock_sm,
      ),
      mock.patch(f"{_MODULE}._secret_exists", return_value=False),
      mock.patch(f"{_MODULE}.click.prompt", side_effect=["myuser", "mytoken"]),
    ):
      _ensure_dockerhub_secrets("proj")

    self.assertEqual(mock_sm.create_secret.call_count, 2)
    self.assertEqual(mock_sm.add_secret_version.call_count, 2)

  def test_force_update_prompts_even_when_exists(self):
    mock_sm = MagicMock()
    with (
      mock.patch(
        f"{_MODULE}.secretmanager.SecretManagerServiceClient",
        return_value=mock_sm,
      ),
      mock.patch(f"{_MODULE}._secret_exists", return_value=True),
      mock.patch(
        f"{_MODULE}.click.prompt", side_effect=["newuser", "newtoken"]
      ),
    ):
      _ensure_dockerhub_secrets("proj", force_update=True)

    # Should NOT create (already exists), but should add new version.
    mock_sm.create_secret.assert_not_called()
    self.assertEqual(mock_sm.add_secret_version.call_count, 2)

  def test_token_prompt_is_hidden(self):
    mock_sm = MagicMock()
    with (
      mock.patch(
        f"{_MODULE}.secretmanager.SecretManagerServiceClient",
        return_value=mock_sm,
      ),
      mock.patch(f"{_MODULE}._secret_exists", return_value=False),
      mock.patch(
        f"{_MODULE}.click.prompt", side_effect=["user", "tok"]
      ) as mock_prompt,
    ):
      _ensure_dockerhub_secrets("proj")

    # First call (username): hide_input=False, second (token): True.
    username_call, token_call = mock_prompt.call_args_list
    self.assertFalse(username_call.kwargs.get("hide_input", False))
    self.assertTrue(token_call.kwargs["hide_input"])


class TestBuildBaseCommand(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_build_failure_reported_gracefully(self):
    with (
      mock.patch(f"{_MODULE}._ensure_dockerhub_secrets"),
      mock.patch(
        f"{_MODULE}.build_and_push_prebuilt_image",
        side_effect=RuntimeError("Cloud Build exploded"),
      ),
    ):
      result = self.runner.invoke(
        build_base,
        ["--project", "proj", "--repo", "r", "--category", "gpu", "-y"],
      )

    # Command should still exit 0 (failures are reported, not fatal).
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("Cloud Build exploded", result.output)
    self.assertIn("Failed", result.output)

  def test_project_required_in_non_interactive_mode(self):
    """When --repo is given (non-interactive), --project must be set."""
    result = self.runner.invoke(
      build_base,
      ["--repo", "r", "-y"],
    )
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("--project", result.output)

  def test_empty_categories_exits_early(self):
    """If no categories are selected, the command exits without building."""
    with (
      mock.patch(f"{_MODULE}.resolve_project", return_value="proj"),
      mock.patch(f"{_MODULE}._prompt_registry_type", return_value="docker-hub"),
      mock.patch(f"{_MODULE}._prompt_repo", return_value="myrepo"),
      mock.patch(f"{_MODULE}._prompt_categories", return_value=[]),
      mock.patch(
        f"{_MODULE}.build_and_push_prebuilt_image",
      ) as mock_build,
    ):
      result = self.runner.invoke(
        build_base,
        ["-y"],
        input="0.0.1\n",
      )

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn("nothing to build", result.output)
    mock_build.assert_not_called()


class TestIsArRepo(absltest.TestCase):
  def test_detects_ar_uri(self):
    self.assertTrue(_is_ar_repo("us-docker.pkg.dev/proj/my-repo"))
    self.assertTrue(_is_ar_repo("europe-docker.pkg.dev/p/r"))

  def test_rejects_dockerhub(self):
    self.assertFalse(_is_ar_repo("kinetic"))
    self.assertFalse(_is_ar_repo("mycompany/kinetic"))


class TestParseArRepo(absltest.TestCase):
  def test_parses_standard_uri(self):
    location, project, repo_name = _parse_ar_repo(
      "us-docker.pkg.dev/my-project/kinetic-base"
    )
    self.assertEqual(location, "us")
    self.assertEqual(project, "my-project")
    self.assertEqual(repo_name, "kinetic-base")

  def test_parses_multipart_location(self):
    location, project, repo_name = _parse_ar_repo(
      "europe-west1-docker.pkg.dev/proj/repo"
    )
    self.assertEqual(location, "europe-west1")


class TestBuildBaseArtifactRegistry(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.runner = CliRunner()

  def test_ar_repo_skips_dockerhub_secrets(self):
    with (
      mock.patch(f"{_MODULE}._ensure_dockerhub_secrets") as mock_ensure,
      mock.patch(
        f"{_MODULE}.build_and_push_prebuilt_image",
        return_value="us-docker.pkg.dev/proj/r/base-gpu:0.0.1",
      ),
    ):
      result = self.runner.invoke(
        build_base,
        [
          "--project",
          "proj",
          "--repo",
          "us-docker.pkg.dev/proj/r",
          "--category",
          "gpu",
          "-y",
        ],
      )

    self.assertEqual(result.exit_code, 0, result.output)
    mock_ensure.assert_not_called()

  def test_update_credentials_with_ar_repo_is_noop(self):
    with (
      mock.patch(f"{_MODULE}._ensure_dockerhub_secrets") as mock_ensure,
      mock.patch(
        f"{_MODULE}.build_and_push_prebuilt_image",
        return_value="us-docker.pkg.dev/proj/r/base-gpu:0.0.1",
      ),
    ):
      result = self.runner.invoke(
        build_base,
        [
          "--project",
          "proj",
          "--repo",
          "us-docker.pkg.dev/proj/r",
          "--category",
          "gpu",
          "--update-credentials",
          "-y",
        ],
      )

    self.assertEqual(result.exit_code, 0, result.output)
    mock_ensure.assert_not_called()
    self.assertIn("no effect", result.output)


if __name__ == "__main__":
  absltest.main()
