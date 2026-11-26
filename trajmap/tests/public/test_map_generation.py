import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "construction" / "generate_map.py"


class MapGenerationTests(unittest.TestCase):
    def test_with_arrows_runs_cloud_clustering_and_publishes_one_line_layer(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        entry = self.load_entry()
        points = np.random.RandomState(4).uniform([-0.25, -0.25, 0], [0.25, 0.25, 0], (1000, 3))
        lanes = [(np.array([[-10., y, 0], [10., y, 0]]), "solid", i)
                 for i, y in enumerate([-1.5, 1.5])]

        class FittedCore:
            def process_lane_segment(self, directory, segment_id, **kwargs):
                kwargs["arrow_processor"](lanes, points, segment_id, np.array([1., 0, 0]), fitting_type=0)
                return lanes, 0, 2

            def save_segment_shapefile_with_numbers(self, lanes, mode, segment_id, count, prefix, **kwargs):
                return gpd.GeoDataFrame({"type": ["solid", "solid"], "id": [0, 1],
                                        "segment_id": [segment_id, segment_id]},
                                       geometry=[LineString(lane[0][:, :2]) for lane in lanes],
                                       crs=kwargs["output_crs"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lane_segment_001.pcd").touch()
            (root / "lane_segment_001_poses.txt").write_text("0 0 0 0 0 0 0 1\n1 60 0 0 0 0 0 1\n")
            output = root / "output"
            with patch.object(entry, "load_core", return_value=FittedCore()):
                code = entry.main(["--segment-dir", str(root), "--input-crs", "EPSG:32650",
                                   "--output-dir", str(output), "--with-arrows"])
            self.assertEqual(code, 0)
            frame = gpd.read_file(output / "vector_map.shp")
            self.assertEqual(list(frame["type"]), ["lane", "lane", "arrow"])
            self.assertTrue(frame.geometry.iloc[-1].is_ring)
            self.assertTrue(frame.edge_id.is_unique)
            self.assertTrue(frame.id.is_unique)
            np.testing.assert_allclose(frame.iloc[-1][["cluster_x", "cluster_y", "cluster_z"]].astype(float),
                                       points.mean(axis=0))
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["arrow_features"], 1)
            self.assertEqual(manifest["lane_features"], 2)
            self.assertEqual(manifest["segments"][0]["arrow_features"], 1)
            self.assertNotIn("arrow_rectangle_export", manifest["missing_stages"])
            self.assertFalse((output / "lane_map.shp").exists())

    def test_failed_arrow_stage_never_publishes_an_incomplete_vector_map(self):
        entry = self.load_entry()

        class InvalidArrowCore:
            def process_lane_segment(self, directory, segment_id, **kwargs):
                lanes = [(np.array([[-5., 1., 0.], [5., 1., 0.]]), "solid", 0)]
                kwargs["arrow_processor"](lanes, np.full((1000, 3), np.nan),
                                          segment_id, np.array([1., 0., 0.]))
                raise AssertionError("Invalid arrow data was silently accepted")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lane_segment_001.pcd").touch()
            (root / "lane_segment_001_poses.txt").write_text("0 0 0 0 0 0 0 1\n1 60 0 0 0 0 0 1\n")
            output = root / "output"
            with patch.object(entry, "load_core", return_value=InvalidArrowCore()):
                status = entry.main(["--segment-dir", str(root), "--input-crs", "EPSG:32650",
                                     "--output-dir", str(output), "--with-arrows"])
            self.assertEqual(status, 1)
            self.assertEqual(list(output.glob("*.shp")), [])
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["segments"][0]["status"], "failed")
            self.assertNotIn("silently accepted", manifest["segments"][0]["error"])

    def test_arrow_export_is_closed_linework_with_separate_cluster_center(self):
        from shapely.geometry import Polygon
        entry = self.load_entry()
        self.assertTrue(hasattr(entry, "arrow_frame"), "Arrow export must be available")
        convert, crs, _ = entry.coordinate_functions("EPSG:32650", None)
        data = [{"bbox_vertices": np.array([[0, 0, 2], [4, 0, 2], [4, 2, 2], [0, 2, 2]]),
                 "center": np.array([1, 0.8, 2]), "label": 4, "group_id": 2}]
        frame = entry.arrow_frame(data, 7, convert, crs, start_id=10)
        self.assertEqual(frame.iloc[0]["type"], "arrow")
        self.assertEqual(frame.iloc[0]["id"], 10)
        self.assertEqual(frame.iloc[0]["segment_id"], 7)
        self.assertEqual(frame.iloc[0]["arrow_id"], 4)
        self.assertEqual(frame.iloc[0]["group_id"], 2)
        geometry = frame.geometry.iloc[0]
        self.assertEqual(geometry.geom_type, "LineString")
        self.assertTrue(geometry.is_ring)
        self.assertFalse(geometry.has_z)
        self.assertAlmostEqual(Polygon(geometry).area, 8)
        self.assertAlmostEqual(frame.iloc[0]["cluster_x"], 1)
        self.assertAlmostEqual(frame.iloc[0]["cluster_y"], 0.8)
        self.assertNotAlmostEqual(Polygon(geometry).centroid.x, frame.iloc[0]["cluster_x"])
        with tempfile.TemporaryDirectory() as tmp:
            entry.publish_map(frame, tmp, "arrows.shp")
            import geopandas as gpd
            loaded = gpd.read_file(Path(tmp) / "arrows.shp")
            self.assertTrue(loaded.geometry.iloc[0].is_ring)
            self.assertAlmostEqual(loaded.iloc[0]["cluster_x"], 1)

    def test_arrow_export_converts_vertices_and_cluster_center_from_enu(self):
        entry = self.load_entry()
        self.assertTrue(hasattr(entry, "arrow_frame"))
        convert, crs, _ = entry.coordinate_functions(None, [32.022252, 118.726685, 8.135])
        center = np.array([2., 1., 0.])
        corners = np.array([[0, 0, 0], [4, 0, 0], [4, 2, 0], [0, 2, 0]])
        frame = entry.arrow_frame([dict(bbox_vertices=corners, center=center, label=0, group_id=0)],
                                  1, convert, crs)
        self.assertEqual(frame.crs.to_epsg(), 4326)
        np.testing.assert_allclose(list(frame.geometry.iloc[0].coords)[:4],
                                   [convert(p)[:2] for p in corners])
        np.testing.assert_allclose(frame.iloc[0][["cluster_x", "cluster_y", "cluster_z"]].astype(float),
                                   convert(center))

    def test_arrow_export_rejects_invalid_geometry_instead_of_skipping(self):
        entry = self.load_entry()
        self.assertTrue(hasattr(entry, "arrow_frame"))
        convert, crs, _ = entry.coordinate_functions("EPSG:32650", None)
        for corners in [np.zeros((4, 3)), np.full((4, 3), np.nan),
                        np.array([[0, 0, 0], [2, 2, 0], [0, 2, 0], [2, 0, 0]])]:
            with self.subTest(corners=corners), self.assertRaises(ValueError):
                entry.arrow_frame([dict(bbox_vertices=corners, center=[1, 1, 0], label=0, group_id=0)],
                                  1, convert, crs)

    def test_splitter_requires_explicit_input_and_output_paths(self):
        spec = importlib.util.spec_from_file_location(
            "stopline_cli", ROOT / "construction" / "save_stopline_points.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, "load_point_cloud_with_class") as read_cloud:
            with self.assertRaises(SystemExit) as raised:
                module.main([])
            self.assertEqual(raised.exception.code, 2)
            read_cloud.assert_not_called()

    def load_entry(self):
        self.assertTrue(ENTRY.is_file(), "A runnable construction entry is required")
        spec = importlib.util.spec_from_file_location("generate_map", ENTRY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_projected_coordinates_are_not_treated_as_enu(self):
        entry = self.load_entry()
        convert, crs, distance = entry.coordinate_functions("EPSG:32652", None)
        np.testing.assert_allclose(convert([350000, 4020000, 10]), [350000, 4020000, 10])
        self.assertEqual(crs.to_epsg(), 32652)
        self.assertAlmostEqual(distance([0, 0], [3, 4]), 5)

    def test_enu_origin_and_east_direction(self):
        entry = self.load_entry()
        convert, crs, _ = entry.coordinate_functions(None, [32.022252, 118.726685, 8.135])
        np.testing.assert_allclose(convert([0, 0, 0]), [118.726685, 32.022252, 8.135], atol=1e-7)
        self.assertGreater(convert([10, 0, 0])[0], 118.726685)
        self.assertEqual(crs.to_epsg(), 4326)

    def test_missing_or_nonmetric_frame_is_rejected(self):
        entry = self.load_entry()
        for crs, origin in [(None, None), ("EPSG:4326", None), ("EPSG:2263", None), ("EPSG:32652", [0, 0, 0])]:
            with self.subTest(crs=crs, origin=origin), self.assertRaises(ValueError):
                entry.coordinate_functions(crs, origin)

    def test_existing_output_is_never_deleted(self):
        entry = self.load_entry()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "existing"
            target.mkdir()
            marker = target / "important.txt"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                entry.prepare_output(target)
            self.assertEqual(marker.read_text(), "keep")

    def test_discovery_includes_tail_and_requires_pose_pair(self):
        entry = self.load_entry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stem in ["lane_segment_001", "lane_segment_incomplete_002"]:
                (root / (stem + ".pcd")).touch()
                (root / (stem + "_poses.txt")).touch()
            self.assertEqual(len(entry.discover_segments(root)), 2)
            (root / "lane_segment_003.pcd").touch()
            with self.assertRaises(ValueError):
                entry.discover_segments(root)

    def test_selected_segments_ignore_unselected_broken_pairs(self):
        entry = self.load_entry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lane_segment_001.pcd").touch()
            (root / "lane_segment_009.pcd").touch()
            (root / "lane_segment_009_poses.txt").touch()
            self.assertEqual(len(entry.discover_segments(root, [9])), 1)

    def test_malformed_global_poses_fail_before_creating_output(self):
        entry = self.load_entry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cloud = root / "semantic.pcd"
            cloud.touch()
            poses = root / "poses.txt"
            poses.write_text("0 0 0 0 0 0 0 1\n1 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n")
            output = root / "output"
            status = entry.main(["--pcd", str(cloud), "--poses", str(poses),
                                 "--input-crs", "EPSG:32652", "--output-dir", str(output)])
            self.assertEqual(status, 1)
            self.assertFalse(output.exists())

    def test_invalid_seed_fails_before_creating_output(self):
        entry = self.load_entry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lane_segment_001.pcd").touch()
            (root / "lane_segment_001_poses.txt").touch()
            output = root / "output"
            status = entry.main(["--segment-dir", str(root), "--seed", "-1",
                                 "--input-crs", "EPSG:32652", "--output-dir", str(output)])
            self.assertEqual(status, 1)
            self.assertFalse(output.exists())

    def test_inline_pose_comments_are_not_silently_dropped_downstream(self):
        entry = self.load_entry()
        with tempfile.TemporaryDirectory() as tmp:
            poses = Path(tmp) / "poses.txt"
            poses.write_text("0 0 0 0 0 0 0 1 # sample\n1 1 0 0 0 0 0 1\n")
            with self.assertRaisesRegex(ValueError, "Inline"):
                entry.validate_poses(poses)

    def test_failed_writer_cannot_leave_a_final_shapefile(self):
        entry = self.load_entry()
        self.assertTrue(hasattr(entry, "publish_map"))
        class FailingWriter:
            def to_file(self, path, **kwargs):
                Path(path).write_bytes(b"incomplete driver output")
                raise OSError("driver failed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(OSError):
                entry.publish_map(FailingWriter(), root, "lane_map.shp")
            self.assertEqual(list(root.iterdir()), [])

    def test_partial_and_strict_modes_report_failed_segments(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        entry = self.load_entry()

        class DeterministicCore:
            def process_lane_segment(self, directory, segment_id, **kwargs):
                return ["lane"], 1, 1

            def save_segment_shapefile_with_numbers(self, lanes, mode, segment_id, count, prefix, **kwargs):
                return gpd.GeoDataFrame({"type": ["solid"], "id": [0],
                                        "segment_id": [segment_id]},
                                       geometry=[LineString([(0, 0), (10, 0)])], crs=kwargs["output_crs"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for number in [1, 2]:
                (root / ("lane_segment_%03d.pcd" % number)).touch()
            (root / "lane_segment_001_poses.txt").write_text("0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n")
            (root / "lane_segment_002_poses.txt").write_text("bad trajectory\n")
            for partial in [False, True]:
                output = root / ("partial" if partial else "strict")
                args = ["--segment-dir", str(root), "--input-crs", "EPSG:32652",
                        "--output-dir", str(output)]
                if partial:
                    args.append("--allow-partial")
                with patch.object(entry, "load_core", return_value=DeterministicCore()):
                    status = entry.main(args)
                manifest = json.loads((output / "run_manifest.json").read_text())
                self.assertEqual(status, 2 if partial else 1)
                self.assertEqual(manifest["status"], "partial" if partial else "failed")
                self.assertEqual([row["status"] for row in manifest["segments"]], ["ok", "failed"])
                self.assertIn("Traceback", (output / "generation.log").read_text())
                self.assertFalse((output / "lane_map.shp").exists())
                self.assertEqual((output / "lane_map_partial.shp").exists(), partial)
                self.assertFalse(any(path.name.startswith("export_") for path in output.iterdir()))

    def test_window_export_preserves_branch_and_numeric_lane_ids(self):
        entry = self.load_entry()
        core = entry.load_core()
        line = lambda x, y: np.array([[x, y, 0.0], [x + 10, y, 0.0]])
        lanes = [([line(0, 3), line(10, 3)], "left solid", 2),
                 ([line(0, -3), line(10, -3)], "right solid", 0),
                 ([line(0, 0), [line(10, 0), line(10, 1)]], "dashed", 1)]
        convert, crs, distance = entry.coordinate_functions("EPSG:32652", None)
        frame = core.save_segment_shapefile_with_numbers(
            lanes, 0, 1, 3, None, point_transform=convert,
            output_crs=crs, distance_function=distance, return_frame=True)
        self.assertGreater(len(frame), 4)
        self.assertTrue(all(isinstance(value, (int, np.integer)) for value in frame["id"]))
        self.assertTrue(all(g.is_valid and g.length > 0 for g in frame.geometry))
        dashed = frame[frame["type"] == "dashed"]
        self.assertTrue(all(
            any(tuple(g.coords[0]) == start and tuple(g.coords[-1]) == end
                and (10.0, 0.0) in g.coords
                for g in dashed.geometry)
            for start, end in [((10.0, 0.0), (20.0, 0.0)),
                               ((0.0, 0.0), (20.0, 1.0))]
        ))


if __name__ == "__main__":
    unittest.main()
