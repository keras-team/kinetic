"""E2E tests for remote execution with CPU accelerator.

These tests require a real GCP project with:
- A GKE cluster with a CPU node pool
- Cloud Storage, Cloud Build, and Artifact Registry APIs enabled
- Proper IAM permissions

Set E2E_TESTS=1 to enable.
"""

import os
from unittest import mock

from absl.testing import absltest

import kinetic
from tests.e2e.e2e_utils import skip_unless_e2e


@skip_unless_e2e()
class TestCpuExecution(absltest.TestCase):
  def setUp(self):
    super().setUp()

  def test_simple_function(self):
    """Execute a simple add function remotely and verify the result."""

    @kinetic.run(accelerator="cpu")
    def add(a, b):
      return a + b

    result = add(2, 3)
    self.assertEqual(result, 5)

  def test_complex_return_type(self):
    """Verify complex return types survive serialization roundtrip."""

    @kinetic.run(accelerator="cpu")
    def complex_return():
      return {
        "key": [1, 2, 3],
        "nested": {"a": True, "b": None},
        "tuple": (4, 5),
      }

    result = complex_return()
    self.assertEqual(result["key"], [1, 2, 3])
    self.assertTrue(result["nested"]["a"])
    self.assertIsNone(result["nested"]["b"])
    self.assertEqual(result["tuple"], (4, 5))

  def test_function_that_raises(self):
    """Verify remote exceptions are re-raised locally."""

    @kinetic.run(accelerator="cpu")
    def bad_func():
      raise ValueError("intentional test error")

    with self.assertRaisesRegex(ValueError, "intentional test error"):
      bad_func()

  def test_env_var_propagation(self):
    """Verify captured env vars are available in the remote environment."""
    with mock.patch.dict(os.environ, {"E2E_TEST_VAR": "hello_from_local"}):

      @kinetic.run(
        accelerator="cpu",
        capture_env_vars=["E2E_TEST_VAR"],
      )
      def read_env():
        return os.environ.get("E2E_TEST_VAR")

      result = read_env()
      self.assertEqual(result, "hello_from_local")

  def test_function_with_args_and_kwargs(self):
    """Verify positional and keyword arguments are passed correctly."""

    @kinetic.run(accelerator="cpu")
    def compute(x, y, scale=1.0, offset=0.0):
      return (x + y) * scale + offset

    result = compute(3, 4, scale=2.0, offset=1.0)
    self.assertEqual(result, 15.0)


if __name__ == "__main__":
  absltest.main()
