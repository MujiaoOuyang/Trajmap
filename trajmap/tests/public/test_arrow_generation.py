"""Headless regression tests for the approved media arrow implementation."""
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HAS_DEPS = all(importlib.util.find_spec(name) for name in
               ("open3d", "sklearn", "shapely"))


@unittest.skipUnless(HAS_DEPS, "Arrow generation requires Open3D, scikit-learn and Shapely")
class ArrowGenerationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "construction/arrow_generation.py").is_file(),
                        "The extracted arrow module must exist")
        self.module = importlib.import_module("construction.arrow_generation")
        x, y = np.meshgrid(np.linspace(-.25, .25, 30), np.linspace(-.15, .15, 30))
        self.points = np.column_stack((x.ravel(), y.ravel() + 1.5, np.zeros(x.size)))
        self.points = np.vstack((self.points, np.tile([.2, 1.5, 0.], (200, 1))))
        self.lanes = [(np.array([[-20., y, 0.], [20., y, 0.]]), kind, i)
                      for i, (y, kind) in enumerate(((0., "right solid"), (3., "dashed")))]

    def run_arrows(self, points=None, lanes=None, direction=(1., 0., 0.), fitting_type=1):
        return self.module.deal_arrow_process(
            self.lanes if lanes is None else lanes,
            self.points if points is None else points, 9, direction, fitting_type=fitting_type)

    def test_dense_cluster_uses_mean_center_and_separate_bbox_midpoint(self):
        original = self.points.copy()
        arrows = self.run_arrows()
        self.assertIsInstance(arrows, list)
        self.assertEqual(len(arrows), 1)
        info = arrows[0]
        self.assertEqual(set(info), {"bbox_vertices", "center", "label", "group_id"})
        self.assertEqual(info["bbox_vertices"].shape, (4, 3))
        np.testing.assert_allclose(info["center"], self.points.mean(axis=0))
        self.assertGreater(np.linalg.norm(info["center"] - info["bbox_vertices"].mean(axis=0)), .03)
        self.assertEqual((info["label"], info["group_id"]), (0, 0))
        np.testing.assert_array_equal(self.points, original)

    def test_rotated_box_follows_vehicle_direction_not_pca(self):
        theta = np.deg2rad(37.)
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                             [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
        plain = self.run_arrows()[0]
        lanes = [(line @ rotation.T, kind, i) for line, kind, i in self.lanes]
        result = self.run_arrows(self.points @ rotation.T, lanes, rotation[:, 0])[0]
        np.testing.assert_allclose(result["bbox_vertices"], plain["bbox_vertices"] @ rotation.T,
                                   atol=1e-12)
        np.testing.assert_allclose(result["center"], plain["center"] @ rotation.T, atol=1e-12)
        self.assertGreater(np.linalg.norm(result["bbox_vertices"][0] - result["bbox_vertices"][1]), .99)

    def test_fitting_type_zero_bypasses_only_half_metre_point_filter(self):
        near = self.points - [0., 1.3, 0.]
        self.assertEqual(self.run_arrows(near, fitting_type=1), [])
        self.assertEqual(len(self.run_arrows(near, fitting_type=0)), 1)
        far = self.points + [0., 10., 0.]
        self.assertEqual(self.run_arrows(far, fitting_type=0), [])

    def test_empty_no_lanes_noise_and_all_filtered_always_return_lists(self):
        for points, lanes in (([], self.lanes), (np.empty((0, 4)), self.lanes),
                              (self.points, []), (self.points, [([], "dashed", 0)]),
                              (self.points[:799], self.lanes),
                              (self.points, self.lanes[:1])):
            with self.subTest(size=len(points), lanes=len(lanes)):
                self.assertEqual(self.run_arrows(points, lanes), [])

    def test_canonical_nested_window_arrays_match_flat_lane_geometry(self):
        expected = self.run_arrows()[0]
        for depth in (1, 2, 3):
            lanes = []
            for line, kind, index in self.lanes:
                midpoint = (line[0] + line[1]) / 2
                nested = [np.array([line[0], midpoint]), np.array([midpoint, line[1]])]
                for _ in range(depth - 1):
                    nested = [nested]
                lanes.append((nested, kind, index))
            with self.subTest(depth=depth):
                result = self.run_arrows(lanes=lanes)
                self.assertEqual(len(result), 1)
                np.testing.assert_allclose(result[0]["bbox_vertices"], expected["bbox_vertices"])
                np.testing.assert_allclose(result[0]["center"], expected["center"])

    def test_degenerate_direction_and_invalid_required_inputs_fail_explicitly(self):
        for direction in ([0., 0., 0.], [0., 0., 1.], [np.nan, 1., 0.], [1.], [[1., 0.]]):
            with self.subTest(direction=direction), self.assertRaisesRegex(ValueError, "direction"):
                self.run_arrows(direction=direction)
        for points in (np.ones((3, 2)), np.ones(3), [[0., np.nan, 0.]]):
            with self.subTest(points=points), self.assertRaisesRegex(ValueError, "arrow"):
                self.run_arrows(points=points)
        with self.assertRaisesRegex(ValueError, "fitting_type"):
            self.run_arrows(fitting_type=2)

    def test_single_window_acceptance_is_an_explicit_source_parser_difference(self):
        # The original boundary filter skips these lists (2 flattened points < 4).
        # The public adapter intentionally treats them as valid two-point lanes.
        lanes = [([line.copy()], kind, index) for line, kind, index in self.lanes]
        expected = self.run_arrows()[0]
        result = self.run_arrows(lanes=lanes)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0]["bbox_vertices"], expected["bbox_vertices"])
        np.testing.assert_allclose(result[0]["center"], expected["center"])

    def test_boundary_filter_retains_infinite_line_two_metre_rule(self):
        self.assertTrue(self.module.is_arrow_in_lane_boundaries([100., 2., 0.], self.lanes[:1]))
        self.assertFalse(self.module.is_arrow_in_lane_boundaries([0., 2.001, 0.], self.lanes[:1]))

    def test_secondary_dbscan_preserves_stale_bbox_center(self):
        points = self.points - [0., 1.5, 0.]
        points = np.vstack((points, [12., 0., 0.]))
        _, _, center, corners = self.module.create_standardized_bbox(points, [0., 0., 0.],
                                                                      np.array([1., 0., 0.]))
        scale = 1 / (1 + 1e-8)
        self.assertAlmostEqual(center[0], ((-.25 + 12.) / 2) * scale ** 2)
        np.testing.assert_allclose(center, corners.mean(axis=0))
        self.assertAlmostEqual(np.linalg.norm(corners[0] - corners[3]), scale)
        self.assertGreater(center[0] - points.mean(axis=0)[0], 5.)

    def test_direction_grouping_keeps_parallel_and_antiparallel_together(self):
        first = self.points - [0., 1.5, 0.]
        perpendicular = first[:, [1, 0, 2]]
        groups = self.module.group_clusters_by_direction([(first, 0), (-first, 1), (perpendicular, 2)])
        self.assertEqual([[label for _, label in group["clusters"]] for group in groups], [[0, 1], [2]])

    def test_headless_processing_never_calls_visualizers(self):
        import open3d as o3d
        with patch.object(o3d.visualization, "draw_geometries", side_effect=AssertionError("GUI")), \
                patch.object(o3d.visualization, "Visualizer", side_effect=AssertionError("GUI")):
            self.assertEqual(len(self.run_arrows()), 1)
            self.assertEqual(self.run_arrows(self.points - [0., 1.3, 0.]), [])

    def test_import_has_no_source_directory_or_gui_dependency(self):
        code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('open3d', 'matplotlib', 'process_lane_segments'):
            raise AssertionError('Import-time dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from construction.arrow_generation import deal_arrow_process
assert deal_arrow_process([], [], 0, [1, 0, 0]) == []
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                env=dict(os.environ, PYTHONPATH=str(ROOT)),
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
