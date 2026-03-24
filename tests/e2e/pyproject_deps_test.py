"""E2E tests for pyproject.toml dependency support.

These tests verify that dependencies declared in ``pyproject.toml`` are
correctly extracted, used for container building, and available in the
remote environment.

A temporary ``pyproject.toml`` is written to a temp directory.  Only the
discovery function (``_find_requirements``) is patched to return that path;
the rest of the pipeline — parsing, JAX filtering, container building, and
remote execution — runs for real.

Set E2E_TESTS=1 to enable.
"""

import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest

import kinetic
from tests.e2e.e2e_utils import skip_unless_e2e


def _make_test_dir(test_case):
  """Create a temp directory cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


@skip_unless_e2e()
class TestPyprojectTomlDependencies(absltest.TestCase):
  """Verify that [project.dependencies] from pyproject.toml are installed."""

  def _create_pyproject(self, content):
    """Write a pyproject.toml in a temp directory and return its path."""
    tmp = _make_test_dir(self)
    pyproject = tmp / "pyproject.toml"
    pyproject.write_text(content)
    return str(pyproject)

  def test_dependency_installed_on_remote(self):
    """A dependency from pyproject.toml is importable in the remote function."""
    path = self._create_pyproject(
      '[project]\nname = "test"\nversion = "0.1"\n'
      'dependencies = ["humanize>=4.0"]\n'
    )

    @kinetic.run(accelerator="cpu")
    def use_humanize():
      import humanize

      return humanize.intcomma(1_000_000)

    with mock.patch(
      "kinetic.backend.execution._find_requirements",
      return_value=path,
    ):
      result = use_humanize()

    self.assertEqual(result, "1,000,000")

  def test_pyproject_without_deps_succeeds(self):
    """A pyproject.toml with no [project.dependencies] doesn't break the pipeline."""
    path = self._create_pyproject("[tool.ruff]\nline-length = 88\n")

    @kinetic.run(accelerator="cpu")
    def simple_add(a, b):
      return a + b

    with mock.patch(
      "kinetic.backend.execution._find_requirements",
      return_value=path,
    ):
      result = simple_add(10, 20)

    self.assertEqual(result, 30)

  def test_jax_filtered_from_pyproject_deps(self):
    """JAX packages in pyproject.toml are filtered like in requirements.txt."""
    path = self._create_pyproject(
      '[project]\nname = "test"\nversion = "0.1"\n'
      'dependencies = ["jax", "humanize>=4.0"]\n'
    )

    @kinetic.run(accelerator="cpu")
    def check_humanize():
      import humanize

      return humanize.intcomma(2_500)

    with mock.patch(
      "kinetic.backend.execution._find_requirements",
      return_value=path,
    ):
      result = check_humanize()

    # humanize was installed (not filtered), jax was filtered silently
    self.assertEqual(result, "2,500")


if __name__ == "__main__":
  absltest.main()
