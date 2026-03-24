"""Integration tests for the full serialization roundtrip."""

import pathlib
import sys
import tempfile
import zipfile

from absl.testing import absltest

from kinetic.utils.packager import zip_working_dir


def _make_temp_path(test_case):
  """Create a temp directory that is cleaned up after the test."""
  td = tempfile.TemporaryDirectory()
  test_case.addCleanup(td.cleanup)
  return pathlib.Path(td.name)


class TestZipAndExtract(absltest.TestCase):
  def _zip_and_extract(self, src, tmp_path):
    """Zip a directory and extract it, returning the extraction path."""
    zip_path = str(tmp_path / "context.zip")
    zip_working_dir(str(src), zip_path)
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
      zf.extractall(str(extract_dir))
    return extract_dir

  def test_zip_extract_preserves_files(self):
    """Zip -> extract roundtrip preserves file content."""
    tmp_path = _make_temp_path(self)
    src = tmp_path / "project"
    src.mkdir()
    (src / "main.py").write_text("x = 1")
    (src / "config.json").write_text('{"key": "val"}')

    extract_dir = self._zip_and_extract(src, tmp_path)

    self.assertEqual((extract_dir / "main.py").read_text(), "x = 1")
    self.assertEqual(
      (extract_dir / "config.json").read_text(), '{"key": "val"}'
    )

  def test_zip_extract_enables_imports(self):
    """Extracted workspace can be added to sys.path for imports."""
    tmp_path = _make_temp_path(self)
    src = tmp_path / "project"
    src.mkdir()
    (src / "helper.py").write_text("def greet(name):\n  return f'Hi {name}'")

    extract_dir = self._zip_and_extract(src, tmp_path)

    sys.path.insert(0, str(extract_dir))
    try:
      import helper

      self.assertEqual(helper.greet("World"), "Hi World")
    finally:
      sys.path.remove(str(extract_dir))
      sys.modules.pop("helper", None)


if __name__ == "__main__":
  absltest.main()
