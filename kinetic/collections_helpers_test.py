"""Tests for kinetic.collections_helpers — input expansion and manifest helpers."""

from absl.testing import absltest, parameterized

from kinetic.collections_helpers import (
  append_child_to_manifest,
  build_initial_manifest,
  call_with_input,
  is_valid_kwargs_dict,
)


def _capture(*args, **kwargs):
  """Return (args, kwargs) so callers can inspect how they were called."""
  return args, kwargs


class TestIsValidKwargsDict(parameterized.TestCase):
  @parameterized.parameters(
    ({"lr": 0.1, "wd": 0.01}, True),
    ({}, True),
    ([1, 2, 3], False),
    ({1: "a", 2: "b"}, False),
    ({"not-an-id": 1}, False),
    ({"class": 1, "name": 2}, False),
  )
  def test_is_valid_kwargs_dict(self, item, expected):
    self.assertEqual(is_valid_kwargs_dict(item), expected)


class TestCallWithInput(parameterized.TestCase):
  @parameterized.named_parameters(
    ("auto_dict_kwargs", {"a": 1, "b": 2}, "auto", (), {"a": 1, "b": 2}),
    ("auto_dict_non_id_single", {"not-id": 1}, "auto", ({"not-id": 1},), {}),
    ("auto_dict_keyword_single", {"class": 1}, "auto", ({"class": 1},), {}),
    ("auto_list_args", [1, 2, 3], "auto", (1, 2, 3), {}),
    ("auto_tuple_args", (10, 20), "auto", (10, 20), {}),
    ("auto_scalar_single", 42, "auto", (42,), {}),
    ("single_mode", [1, 2], "single", ([1, 2],), {}),
    ("args_mode", [10, 20], "args", (10, 20), {}),
    ("kwargs_mode", {"x": 1}, "kwargs", (), {"x": 1}),
  )
  def test_dispatch(self, item, mode, expected_args, expected_kwargs):
    result = call_with_input(_capture, item, mode)
    self.assertEqual(result, (expected_args, expected_kwargs))

  @parameterized.parameters(
    (42, "args", TypeError),
    ([1, 2], "kwargs", TypeError),
    (1, "bogus", ValueError),
  )
  def test_dispatch_errors(self, item, mode, error_type):
    with self.assertRaises(error_type):
      call_with_input(_capture, item, mode)


class TestManifestHelpers(absltest.TestCase):
  def test_build_initial_manifest(self):
    m = build_initial_manifest(
      "grp-1", "map", "my-batch", {"k": "v"}, 10, "train"
    )
    self.assertEqual(m["group_id"], "grp-1")
    self.assertEqual(m["group_kind"], "map")
    self.assertEqual(m["group_name"], "my-batch")
    self.assertEqual(m["tags"], {"k": "v"})
    self.assertEqual(m["total_expected"], 10)
    self.assertEqual(m["submit_fn_name"], "train")
    self.assertEqual(m["children"], [])
    self.assertIn("created_at", m)

  def test_build_initial_manifest_defaults(self):
    m = build_initial_manifest("grp-2", "map", None, None, 5, "fn")
    self.assertIsNone(m["group_name"])
    self.assertEqual(m["tags"], {})

  def test_append_child_new(self):
    m = {"children": []}
    append_child_to_manifest(m, 0, "job-a")
    self.assertEqual(len(m["children"]), 1)
    self.assertEqual(m["children"][0]["group_index"], 0)
    self.assertEqual(m["children"][0]["job_id"], "job-a")
    self.assertEqual(m["children"][0]["attempts"], 1)

  def test_append_child_updates_existing(self):
    m = {
      "children": [
        {"group_index": 0, "job_id": "job-old", "attempts": 1},
      ]
    }
    append_child_to_manifest(m, 0, "job-new", attempts=2)
    self.assertEqual(len(m["children"]), 1)
    self.assertEqual(m["children"][0]["job_id"], "job-new")
    self.assertEqual(m["children"][0]["attempts"], 2)


if __name__ == "__main__":
  absltest.main()
