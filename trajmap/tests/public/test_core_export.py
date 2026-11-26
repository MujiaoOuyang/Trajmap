import builtins
import contextlib
import importlib.util
import io
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np


CORE = Path(__file__).resolve().parents[2] / "construction" / "process_lane_segments.py"


def load_core():
    spec = importlib.util.spec_from_file_location("core_export_test", CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def line(x, y):
    return np.array([[x, y, 0.0], [x + 10, y, 0.0]])


class CoreExportTests(unittest.TestCase):
    def setUp(self):
        self.core = load_core()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_optional_arrow_callback_preserves_lane_return_contract(self):
        self.assertIn("arrow_processor", inspect.signature(self.core.process_lane_segment).parameters)
        lanes = [(line(0, 2), "solid", 0)]
        points = np.array([[i / 10., 0, 0, 2] for i in range(201)] + [[3, 1, 0, 3]])
        poses = [(0, [0, 0, 0, 0, 0, 0, 1]), (1, [60, 0, 0, 0, 0, 0, 1])]
        calls = []
        def capture(*args, **kwargs):
            calls.append((args, kwargs))
        with patch.object(self.core, "load_point_cloud", return_value=points), \
                patch.object(self.core, "load_poses", return_value=poses), \
                patch.object(self.core, "classify_dashed_lane_with_ends", return_value=(points, 20, 0)), \
                patch.object(self.core, "sliding_window_lane_fitting", return_value=lanes):
            baseline = self.core.process_lane_segment("unused", 1)
            actual = self.core.process_lane_segment("unused", 1, arrow_processor=capture)
        self.assertIs(actual[0], baseline[0])
        self.assertEqual(actual[1:], baseline[1:])
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0][0], lanes)
        np.testing.assert_allclose(calls[0][0][1][:, :3], [[3, 1, 0]])
        np.testing.assert_allclose(calls[0][0][3], [1, 0, 0], atol=1e-7)
        self.assertEqual(calls[0][1]["fitting_type"], 0)

    def export(self, lanes, generated_type=0, **kwargs):
        options = dict(point_transform=lambda p: np.asarray(p) + [100, 200, 0],
                       output_crs="EPSG:32652", distance_function=lambda a, b: 0,
                       return_frame=True)
        options.update(kwargs)
        with patch.object(gpd.GeoDataFrame, "to_file", side_effect=AssertionError("unexpected write")):
            return self.core.save_segment_shapefile_with_numbers(
                lanes, generated_type, 7, 3, None, **options)

    def test_window_branch_and_lane_metadata(self):
        lanes = [([line(0, 3), line(10, 3)], "left solid", 2),
                 ([line(0, -3), line(10, -3)], "right solid", 0),
                 ([line(0, 0), [line(10, 0), line(10, 1)]], "dashed", 1)]
        calls = []

        def distance(a, b):
            calls.append((tuple(a), tuple(b)))
            return 0

        frame = self.export(lanes, distance_function=distance)
        self.assertEqual(frame.crs.to_epsg(), 32652)
        self.assertGreater(len(frame), 4)
        self.assertTrue(all(g.is_valid and g.length > 0 for g in frame.geometry))
        self.assertTrue(all(isinstance(value, (int, np.integer)) for value in frame.id))
        for kind, feature_id, left in [("left solid", 0, 3), ("right solid", 1, 1), ("dashed", 2, 2)]:
            subset = frame[frame["type"] == kind]
            self.assertEqual(set(subset.id), {feature_id})
            self.assertEqual(set(subset.left_lane), {left})
        self.assertEqual(calls, [((110.0, 200.0), (110.0, 200.0))])
        dashed = frame[frame["type"] == "dashed"]
        self.assertTrue(any(tuple(g.coords[0]) == (110, 200) and
                            tuple(g.coords[-1]) == (120, 200)
                            for g in dashed.geometry))
        self.assertTrue(any(tuple(g.coords[0]) == (100, 200) and
                            tuple(g.coords[-1]) == (120, 201) and
                            (110, 200) in g.coords
                            for g in dashed.geometry))

    def test_same_type_features_keep_separate_metadata(self):
        frame = self.export([([line(0, 3)], "left solid", 2),
                             ([line(0, 6)], "left solid", 5)])
        self.assertEqual(list(frame.id), [0, 1])
        self.assertEqual(list(frame.left_lane), [3, 6])

    def test_distance_gate_does_not_add_shared_point_when_disconnected(self):
        lanes = [([line(0, 0), [line(15, 0), line(15, 1)]], "dashed", 1)]
        frame = self.export(lanes, distance_function=lambda a, b: 61)
        self.assertEqual({tuple(g.coords[0]) for g in frame.geometry},
                         {(100, 200), (115, 200), (115, 201)})

    def test_global_export_transforms_all_points_and_skips_zero_length(self):
        points = np.array([[0, 0, 0], [10, 0, 0], [10, 0, 0], [20, 0, 0]])
        for kind in ("solid", "dashed"):
            with self.subTest(kind=kind):
                frame = self.export([(points, kind, 1)], generated_type=1)
                self.assertEqual([list(g.coords) for g in frame.geometry],
                                 [[(100, 200), (110, 200)], [(110, 200), (120, 200)]])

    def test_empty_frame_has_geometry_schema_and_crs(self):
        frame = self.export([])
        self.assertIsInstance(frame, gpd.GeoDataFrame)
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns),
                         ["id", "type", "segment_id", "left_lane", "right_lane", "geometry"])
        self.assertEqual(frame.crs.to_epsg(), 32652)

    def test_frame_mode_propagates_conversion_errors(self):
        def fail(point):
            raise ValueError("conversion failed")
        with self.assertRaisesRegex(ValueError, "conversion failed"):
            self.export([([line(0, 0)], "left solid", 0)], point_transform=fail)

    def test_legacy_defaults_write_frame_and_return_boolean(self):
        written = []
        def capture(frame, path):
            written.append((frame, path))
        with patch.object(self.core, "enu_to_wgs84", side_effect=lambda lat, lon, h, p: p), \
                patch.object(gpd.GeoDataFrame, "to_file", capture):
            result = self.core.save_segment_shapefile_with_numbers(
                [(line(0, 0), "solid", 0)], 1, 7, 1, "legacy")
        self.assertIs(result, True)
        self.assertEqual(written[0][1], "shp_temp/legacy_segment_007_lanes.shp")
        self.assertEqual(written[0][0].crs.to_epsg(), 4326)
        with patch.object(self.core, "enu_to_wgs84", side_effect=ValueError("conversion failed")):
            self.assertIs(self.core.save_segment_shapefile_with_numbers(
                [(line(0, 0), "solid", 0)], 1, 7, 1, "legacy"), False)


class WindowDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.core = load_core()

    def fit_solid_window(self, fit_results):
        points = np.column_stack((np.linspace(0, 10, 600),
                                  np.zeros(600), np.zeros(600), np.ones(600)))
        diagnostics = []

        def capture(points, segments, *args):
            diagnostics.append(segments)

        # Isolate the diagnostic paths from stochastic RANSAC and GUI rendering.
        with patch.object(self.core, "separate_dashed_lanes_by_fitting", side_effect=fit_results), \
                patch.object(self.core, "visualize_single_window", capture), \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.core.sliding_window_lane_fitting(points, np.array([1., 0., 0.]))
        return {kind: segments for segments, kind, _ in result}, diagnostics

    def test_solid_only_single_line_keeps_fitted_geometry(self):
        for y, kind in [(3, "left solid"), (-3, "right solid")]:
            with self.subTest(kind=kind):
                result, diagnostics = self.fit_solid_window([([line(0, y)], [600])])
                self.assertEqual(len(result[kind]), 1)
                np.testing.assert_allclose(result[kind][0], [[1.1, y, 0], [10, y, 0]])
                self.assertIs(diagnostics[0][kind.replace(" ", "_")][0], result[kind][0])

    def test_solid_only_three_lines_use_appended_boundary_references(self):
        result, diagnostics = self.fit_solid_window(
            [([line(0, 3), line(0, 0), line(0, -3)], [600, 600, 600])])
        for y, kind in [(3, "left solid"), (-3, "right solid")]:
            np.testing.assert_allclose(result[kind][0], [[1.1, y, 0], [10, y, 0]])
            self.assertIs(diagnostics[0][kind.replace(" ", "_")][0], result[kind][0])

    def test_solid_only_fallback_diagnostics_preserve_appended_geometry(self):
        result, diagnostics = self.fit_solid_window(
            [([], []), ([line(0, 3), line(0, -3)], [600, 600])])
        for y, kind in [(3, "left solid"), (-3, "right solid")]:
            np.testing.assert_allclose(result[kind][0], [[0, y, 0], [10, y, 0]])
            self.assertIs(diagnostics[0][kind.replace(" ", "_")][0], result[kind][0])

    def test_first_empty_fit_window_has_empty_diagnostics(self):
        result, diagnostics = self.fit_solid_window([([], []), ([], [])])
        self.assertTrue(all(not segments for segments in result.values()))
        self.assertEqual(diagnostics, [{}])


class LazyGeographicLibTests(unittest.TestCase):
    def test_import_does_not_require_geographiclib(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.startswith("geographiclib"):
                raise ImportError("geographiclib intentionally unavailable")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", guarded_import):
            core = load_core()
            with self.assertRaisesRegex(ImportError, "intentionally unavailable"):
                core.enu_to_wgs84(32, 118, 0, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
