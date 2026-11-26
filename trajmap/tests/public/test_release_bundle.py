"""Publication packaging must not sweep local research data into the archive."""

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]


class ReleaseBundleTests(unittest.TestCase):
    def load_builder(self):
        path = ROOT / "scripts" / "build_release.py"
        self.assertTrue(path.is_file(), "The allowlist-based release builder is missing")
        spec = importlib.util.spec_from_file_location("build_release", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_only_allowlisted_files_are_archived(self):
        builder = self.load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("Public documentation\n")
            (root / "private.shp").write_bytes(b"private survey")
            manifest = root / "release_manifest.json"
            manifest.write_text(json.dumps({"files": ["README.md", "release_manifest.json"]}))
            output = root / "dist" / "release.zip"
            builder.build_release(root, manifest, output)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), {
                    "trajmap/README.md",
                    "trajmap/release_manifest.json",
                    "trajmap/SHA256SUMS",
                })
                self.assertIn(b"README.md", archive.read("trajmap/SHA256SUMS"))
            with self.assertRaises(FileExistsError):
                builder.build_release(root, manifest, output)

    def test_rejects_escaping_and_noncanonical_paths(self):
        builder = self.load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for entry in ("../secret", "/etc/passwd", "a/../secret", "a\\secret", "./README.md"):
                manifest = root / "release_manifest.json"
                manifest.write_text(json.dumps({"files": [entry]}))
                with self.subTest(entry=entry), self.assertRaises(ValueError):
                    builder.build_release(root, manifest, root / "release.zip")
                self.assertFalse((root / "release.zip").exists())

    def test_rejects_symlinks_and_missing_files(self):
        builder = self.load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "target.txt").write_text("private")
            (root / "link.txt").symlink_to(root / "target.txt")
            for entry in ("link.txt", "missing.txt"):
                manifest = root / "release_manifest.json"
                manifest.write_text(json.dumps({"files": [entry]}))
                with self.subTest(entry=entry), self.assertRaises((ValueError, FileNotFoundError)):
                    builder.build_release(root, manifest, root / "release.zip")

    def test_rejects_duplicate_manifest_entries(self):
        builder = self.load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("docs")
            manifest = root / "release_manifest.json"
            manifest.write_text(json.dumps({"files": ["README.md", "README.md"]}))
            with self.assertRaises(ValueError):
                builder.build_release(root, manifest, root / "release.zip")

    def test_export_directory_is_upload_ready_and_excludes_private_files(self):
        builder = self.load_builder()
        self.assertTrue(hasattr(builder, "export_directory"), "A clean GitHub upload directory is required")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "code").mkdir()
            (root / "code" / "tool.py").write_text("print('ok')\n")
            (root / "private.pcd").write_bytes(b"private dataset")
            manifest = root / "release_manifest.json"
            manifest.write_text(json.dumps({"files": ["code/tool.py", "release_manifest.json"]}))
            output = root / "dist" / "trajmap"
            files = builder.export_directory(root, manifest, output)
            self.assertEqual(files, ["code/tool.py", "release_manifest.json"])
            actual = sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())
            self.assertEqual(actual, ["SHA256SUMS", "code/tool.py", "release_manifest.json"])
            for line in (output / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                builder.export_directory(root, manifest, output)
            self.assertEqual((output / "code/tool.py").read_text(), "print('ok')\n")

    def test_directory_validation_precedes_output_creation(self):
        builder = self.load_builder()
        self.assertTrue(hasattr(builder, "export_directory"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "release_manifest.json"
            manifest.write_text(json.dumps({"files": ["../private"]}))
            output = root / "upload"
            with self.assertRaises(ValueError):
                builder.export_directory(root, manifest, output)
            self.assertFalse(output.exists())

    def test_repository_manifest_is_complete_and_has_no_dataset_files(self):
        manifest = ROOT / "release_manifest.json"
        self.assertTrue(manifest.is_file(), "The public file manifest is missing")
        files = json.loads(manifest.read_text())["files"]
        for entry in files:
            with self.subTest(entry=entry):
                self.assertTrue((ROOT / entry).is_file())
                self.assertNotIn(Path(entry).suffix.lower(), {
                    ".shp", ".shx", ".dbf", ".pcd", ".bag", ".pdf", ".csv", ".png"
                })
                self.assertNotIn(Path(entry).parts[0], {"paper", "results", ".codex_tmp"})


if __name__ == "__main__":
    unittest.main()
