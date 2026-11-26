"""Synthetic metric tests, independent of the private manuscript dataset."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = ROOT / "evaluation_trajmap/geometry/local/evaluate_vector_geometry.py"
HAS_GIS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "geopandas", "shapely", "pyproj"))


@unittest.skipUnless(HAS_GIS, "Install requirements.txt to run the geometry tests")
class GeometryMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("public_geometry", GEOMETRY)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def feature(self, name, points, lane_type="solid"):
        import numpy as np
        points = np.asarray(points, dtype=float)
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        return self.module.Feature(name, "lane", lane_type, points, length)

    def test_spacing_is_an_upper_bound_with_endpoints(self):
        import numpy as np
        from shapely.geometry import LineString
        points = self.module.sample_line(LineString([(0, 0), (1.05, 0)]), 0.2)
        self.assertEqual(len(points), 7)
        np.testing.assert_allclose(points[[0, -1]], [[0, 0], [1.05, 0]])
        np.testing.assert_allclose(np.diff(points[:, 0]), 0.175)

    def test_unequal_point_counts_and_asymmetric_distances(self):
        gt = self.feature("gt", [(0, 0), (1, 0)])
        pred = self.feature("pred", [(0, 0)])
        result = self.module.global_chamfer_metrics([gt], [pred], [0.2])
        self.assertEqual(result["gt_to_pred"]["mean_m"], 0.5)
        self.assertEqual(result["pred_to_gt"]["mean_m"], 0.0)
        self.assertEqual(result["symmetric_cd_m"], 0.5)
        self.assertEqual(result["gt_to_pred"]["coverage_within_0.2m"], 0.5)
        self.assertEqual(result["pred_to_gt"]["coverage_within_0.2m"], 1.0)

    def test_overall_matching_can_be_better_than_each_semantic_class(self):
        low = [(0, 0), (1, 0)]
        high = [(0, 3), (1, 3)]
        gt_s = self.feature("gt_s", low, "solid")
        gt_d = self.feature("gt_d", high, "dashed")
        pred_s = self.feature("pred_s", high, "solid")
        pred_d = self.feature("pred_d", low, "dashed")
        overall = self.module.global_chamfer_metrics([gt_s, gt_d], [pred_s, pred_d], [0.2])
        solid = self.module.global_chamfer_metrics([gt_s], [pred_s], [0.2])
        dashed = self.module.global_chamfer_metrics([gt_d], [pred_d], [0.2])
        self.assertEqual(overall["symmetric_cd_m"], 0.0)
        self.assertEqual(solid["symmetric_cd_m"], 6.0)
        self.assertEqual(dashed["symmetric_cd_m"], 6.0)

    def test_rotated_rectangle_linework_and_iou(self):
        from shapely.affinity import rotate, translate
        from shapely.geometry import LineString, box
        rectangle = rotate(box(0, 0, 2, 4), 37)
        polygon = self.module.linework_to_polygon(LineString(rectangle.exterior.coords))
        self.assertAlmostEqual(self.module.rotated_iou(rectangle, polygon), 1.0)
        self.assertEqual(self.module.rotated_iou(rectangle, translate(polygon, xoff=20)), 0.0)

    def test_geometry_cli_on_synthetic_shapefiles(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, offset in (("gt", 0.0), ("pred", 0.1)):
                gdf = gpd.GeoDataFrame(
                    {"type": ["lane", "arrow"], "lane_type": ["solid", "unknown"]},
                    geometry=[
                        LineString([(500000, offset), (500010, offset)]),
                        LineString([(500000, 5 + offset), (500002, 5 + offset),
                                    (500002, 9 + offset), (500000, 9 + offset), (500000, 5 + offset)]),
                    ], crs="EPSG:32650",
                )
                gdf.to_file(root / f"{name}.shp", index=False)
            result = subprocess.run([
                sys.executable, "-B", str(GEOMETRY), "--gt", "gt.shp", "--pred", "pred.shp",
                "--sample-spacing", "0.2", "--output-json", "metrics.json", "--output-csv", "metrics.csv",
            ], cwd=root, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((root / "metrics.json").read_text())
            lane = report["lane_overall"]["global_chamfer"]
            self.assertAlmostEqual(lane["gt_to_pred"]["mean_m"], 0.1)
            self.assertAlmostEqual(lane["symmetric_cd_m"], 0.2)
            self.assertEqual(lane["gt_to_pred"]["coverage_within_0.2m"], 1.0)
            self.assertEqual(report["arrow"]["num_gt"], 1)
            self.assertEqual(report["arrow"]["num_pred"], 1)
            self.assertTrue((root / "metrics.csv").is_file())

    def test_local_topology_cli_on_identical_synthetic_maps(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        script = ROOT / "evaluation_trajmap/topology/evaluate_topology_metrics.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gdf = gpd.GeoDataFrame(
                {"type": ["lane", "lane"], "lane_type": ["solid", "dashed"]},
                geometry=[LineString([(500000, y), (500040, y)]) for y in (0, 4)],
                crs="EPSG:32650",
            )
            for name in ("gt", "pred"):
                gdf.to_file(root / f"{name}.shp", index=False)
            result = subprocess.run([
                sys.executable, "-B", str(script), "--gt", "gt.shp", "--pred", "pred.shp",
                "--segmentation-mode", "line", "--segment-lengths", "5", "10", "20",
                "--primary-segment-length", "10", "--angle-threshold", "45",
                "--output-json", "metrics.json", "--output-csv", "metrics.csv", "--output-plot", "",
            ], cwd=root, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((root / "metrics.json").read_text())
            self.assertEqual(report["primary"]["segment_length_m"], 10.0)
            self.assertEqual(len(report["segment_length_results"]), 3)
            for item in report["segment_length_results"]:
                self.assertAlmostEqual(item["local_topology_f1"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
