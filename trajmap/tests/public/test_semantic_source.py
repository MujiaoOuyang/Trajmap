"""Lightweight source-packaging checks; no models, real data, or GUI runs."""

import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import runpy

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "construction" / "semantic"


class SemanticSourceTests(unittest.TestCase):
    def test_bundled_inference_config_is_self_contained(self):
        path = ROOT / "construction/semantic/configs/segformer_whu_inference.py"
        self.assertTrue(path.is_file(), "The inference model configuration must be included")
        config = runpy.run_path(str(path))
        self.assertNotIn("_base_", config)
        model = config["model"]
        self.assertEqual(model["backbone"]["num_layers"], [3, 4, 6, 3])
        self.assertEqual(model["decode_head"]["num_classes"], 19)
        self.assertEqual(model["test_cfg"]["crop_size"], (512, 1024))
        self.assertEqual(model["test_cfg"]["stride"], (768, 768))
        self.assertIsNone(model["backbone"]["init_cfg"])
        self.assertEqual(config["test_pipeline"][1]["scale"], (2048, 1024))
        self.assertEqual(config["test_dataloader"]["dataset"]["pipeline"], config["test_pipeline"])
        self.assertNotIn("data_root", config)

    def module(self, name):
        self.assertTrue((SOURCE / (name + ".py")).is_file())
        return importlib.import_module("construction.semantic." + name)

    def isolated(self, body):
        script = """
import importlib.abc, runpy, sys
class NoOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'mmseg', 'mmcv', 'mmengine', 'detectron2', 'imseg',
            'mask2former', 'open3d', 'pclpy', 'pypcd4', 'cv2', 'sklearn',
            'scipy', 'matplotlib'
        }:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, NoOptionalImports())
""" + body
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT))
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp,
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])
        return result

    def test_imports_do_not_load_optional_backends_or_process_files(self):
        self.assertTrue(SOURCE.is_dir())
        self.isolated("""
import importlib
for name in ('myConcateTemp', 'predict', 'arrow_postprocess', 'arrow_vectorized'):
    importlib.import_module('construction.semantic.' + name)
assert 'pypcd_imp' not in sys.modules
""")

    def test_cli_help_without_optional_backends(self):
        for name in ("myConcateTemp", "arrow_vectorized"):
            with self.subTest(name=name):
                self.assertTrue((SOURCE / (name + ".py")).is_file())
                result = self.isolated(
                    "sys.argv = ['tool', '--help']\n"
                    "runpy.run_module('construction.semantic.%s', run_name='__main__')\n" % name)
                self.assertIn("usage:", result.stdout)

    def test_generation_requires_explicit_paths_and_calibration(self):
        module = self.module("myConcateTemp")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            module.main([])
        self.assertEqual(error.exception.code, 2)

    def test_calibration_controls_projection_without_changing_collision_rule(self):
        module = self.module("myConcateTemp")
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "calibration.json"
            config.write_text(json.dumps({"intrinsic": np.eye(3).tolist(),
                                          "extrinsic_r2c": np.eye(4).tolist()}))
            module.configure_calibration(config)
        points = np.array([[2., 2., 2., 7.], [4., 4., 4., 9.],
                           [0., 0., 1., 3.], [0., 0., -1., 4.]])
        with contextlib.redirect_stdout(io.StringIO()):
            depth = module.pcl2image(points, 4, 4, np.eye(4))
            cloud = module.img2pcl(depth)
        # The original loop keeps the last point at a pixel, not nearest depth.
        np.testing.assert_array_equal(depth[1, 1], [9., 4., 0.])
        self.assertEqual(np.count_nonzero(depth[:, :, 1]), 1)
        np.testing.assert_array_equal(cloud, [[4., 4., 4., 0.]])

    def test_sparse_arrow_projection_preserves_class_and_rounding(self):
        module = self.module("myConcateTemp")
        module.K = np.eye(3)
        module.extrinsic_r2c = np.eye(4)
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 2] = True
        depth = np.zeros((3, 3, 3))
        depth[1, 2, 1] = 2.126
        result = module.project_arrows_to_lidar(mask, depth, np.empty((0, 4)), dense_sampling=False)
        np.testing.assert_array_equal(result, [[4.25, 2.13, 2.13, 3.]])

    def test_arrow_helpers_preserve_geometric_and_class_semantics(self):
        module = self.module("arrow_postprocess")
        self.assertEqual(module.compute_line_intersection([0, 0, 4, 4], [0, 4, 4, 0]), (2, 2))
        self.assertIsNone(module.compute_line_intersection([0, 0, 4, 0], [0, 1, 4, 1]))
        self.assertEqual(module.estimate_vanishing_point([], (90, 120)), (60, 30))
        np.testing.assert_array_equal(module.get_lane_mask(np.array([[1, 2, 3]])), [[255, 0, 0]])
        vector = self.module("arrow_vectorized")
        np.testing.assert_array_equal(vector.filter_points_by_class(
            np.array([[1, 2, 3, 3], [4, 5, 6, 1]]), [3]), [[1, 2, 3, 3]])

    def test_colormaps_are_available_without_inference(self):
        module = self.module("predict")
        np.testing.assert_array_equal(module.get_colors("cityscapes")[0], [128, 64, 128])
        np.testing.assert_array_equal(module.get_colors("mapillary")[0], [165, 42, 42])

    def test_lidar_reader_uses_pypcd4_fields_without_aliases(self):
        module = self.module("myConcateTemp")
        self.assertTrue(hasattr(module, "read_lidar_points"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.pcd"
            path.write_text(
                "VERSION .7\nFIELDS x y z intensity ring timestamp\n"
                "SIZE 4 4 4 4 2 8\nTYPE F F F F U F\nCOUNT 1 1 1 1 1 1\n"
                "WIDTH 2\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\nDATA ascii\n"
                "1 2 3 4 5 6\n7 8 9 10 11 12\n")
            np.testing.assert_array_equal(module.read_lidar_points(path), [[1, 2, 3, 4], [7, 8, 9, 10]])
            self.assertNotIn("pypcd_imp", sys.modules)

    def test_invalid_calibration_does_not_replace_active_matrices(self):
        module = self.module("myConcateTemp")
        module.K = np.eye(3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"intrinsic": np.zeros((3, 3)).tolist(),
                                        "extrinsic_r2c": np.eye(4).tolist()}))
            with self.assertRaises(ValueError):
                module.configure_calibration(path)
        np.testing.assert_array_equal(module.K, np.eye(3))

    def test_car_mask_is_explicit_and_recovery_preserves_arrow_region(self):
        module = self.module("arrow_postprocess")
        seg = np.zeros((90, 120), dtype=np.uint8)
        seg[40:50, 50:60] = 3
        with contextlib.redirect_stdout(io.StringIO()):
            recovered = module.extract_arrows_from_segmentation(
                {"predictions": seg}, car_mask=np.ones_like(seg))
            excluded = module.extract_arrows_from_segmentation(
                {"predictions": seg}, car_mask=np.zeros_like(seg))
        np.testing.assert_array_equal(recovered, (seg == 3).astype(np.uint8))
        self.assertEqual(np.count_nonzero(excluded), 0)


if __name__ == "__main__":
    unittest.main()
