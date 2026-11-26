"""Release CLI checks using only temporary, synthetic geospatial inputs."""

import ast
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "evaluation_trajmap/geometry/global/topology_eval.py"
DEPENDENCIES = ("geopandas", "matplotlib", "networkx", "numpy", "scipy", "shapely", "pyproj")
HAS_GIS = all(importlib.util.find_spec(name) is not None for name in DEPENDENCIES)


class GlobalTopologySourceTests(unittest.TestCase):
    def test_import_configuration_has_no_private_paths_or_implicit_crs(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        settings = {
            "TARGET_CRS", "GT_SHP_PATH", "PRED_SHP_PATH", "OUTPUT_DIR",
            "DEBUG_DIR", "GT_WITH_NODES_PATH", "PRED_WITH_NODES_PATH",
        }
        assignments = {
            target.id: node.value
            for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
        }
        for name in settings:
            with self.subTest(setting=name):
                self.assertIsInstance(assignments[name], ast.Constant)
                self.assertIsNone(ast.literal_eval(assignments[name]))

    def test_script_entrypoint_uses_cli_not_unconfigured_main(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        entrypoint = tree.body[-1]
        self.assertIsInstance(entrypoint, ast.If)
        calls = [node.func.id for node in ast.walk(entrypoint)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertIn("cli", calls)
        self.assertNotIn("main", calls)


@unittest.skipUnless(HAS_GIS, "GIS dependencies unavailable; source checks still run")
class GlobalTopologyCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import geopandas as gpd

        # Import must not open a dataset, including on older source versions.
        spec = importlib.util.spec_from_file_location("global_topology_cli_test", SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        with patch.object(gpd, "read_file", side_effect=AssertionError("Import read a dataset")):
            spec.loader.exec_module(cls.module)

    def setUp(self):
        temporary_root = ROOT / ".codex_tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="global-cli-", dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gt = self.root / "gt.shp"
        self.pred = self.root / "pred.shp"
        self.output = self.root / "output"
        settings = {
            "GT_SHP_PATH": self.gt, "PRED_SHP_PATH": self.pred,
            "OUTPUT_DIR": self.output, "DEBUG_DIR": self.output / "debug_reports",
            "GT_WITH_NODES_PATH": self.output / "gt_with_nodes.shp",
            "PRED_WITH_NODES_PATH": self.output / "pred_with_nodes.shp",
            "TARGET_CRS": "EPSG:3857", "MATCH_TOLERANCE": 14.5,
        }
        self.stack = self.enter_context(ExitStack())
        for name, value in settings.items():
            self.stack.enter_context(patch.object(self.module, name, value))
        read_file = self.module.gpd.read_file

        def read_synthetic_only(path, *args, **kwargs):
            resolved = Path(path).resolve()
            if self.root not in resolved.parents:
                raise AssertionError("Test attempted to read a non-fixture dataset")
            return read_file(path, *args, **kwargs)

        self.stack.enter_context(patch.object(self.module.gpd, "read_file", side_effect=read_synthetic_only))

    def enter_context(self, context):
        value = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return value

    def fixture(self, path, crs="EPSG:4326", with_nodes=True, coordinates=None):
        data = {"type": ["lane"]}
        if with_nodes:
            data.update(source=["a"], target=["b"])
        gdf = self.module.gpd.GeoDataFrame(
            data, geometry=[self.module.LineString(
                coordinates if coordinates is not None else [(0, 0), (0.002, 0)]
            )], crs=crs,
        )
        gdf.to_file(path, index=False)
        return path

    def arguments(self, target="EPSG:3857"):
        return ["--gt", str(self.gt), "--pred", str(self.pred),
                "--output-dir", str(self.output), "--target-crs", target]

    def call_cli(self, args):
        self.assertTrue(callable(getattr(self.module, "cli", None)), "CLI is missing")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return self.module.cli(args)

    def assert_cli_error(self, args, message):
        self.assertTrue(callable(getattr(self.module, "cli", None)), "CLI is missing")
        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                self.module.cli(args)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_help_subprocess_without_inputs_or_output(self):
        # Do not execute an old entrypoint that still uses private paths.
        self.assertTrue(callable(getattr(self.module, "cli", None)), "CLI is missing")
        environment = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1",
                           MPLCONFIGDIR=str(self.root / "mpl"))
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--help"], cwd=self.root,
            env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--gt", "--pred", "--output-dir", "--target-crs", "--match-tolerance"):
            self.assertIn(option, result.stdout)
        self.assertIn("diagnostic", result.stdout.lower())
        self.assertFalse(self.output.exists())

    def test_required_arguments_fail_before_io(self):
        self.assert_cli_error([], "required")
        self.assertFalse(self.output.exists())

    def test_missing_input_fails_before_output_creation(self):
        self.assert_cli_error(self.arguments(), "input")
        self.assertFalse(self.output.exists())

    def test_nonmetric_or_invalid_target_crs_is_rejected(self):
        self.fixture(self.gt)
        self.fixture(self.pred)
        for target in ("EPSG:4326", "EPSG:2263", "EPSG:4978", "not-a-crs"):
            with self.subTest(target=target):
                self.assert_cli_error(self.arguments(target), "CRS")
                self.assertFalse(self.output.exists())

    def test_invalid_match_tolerances_are_rejected(self):
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(tolerance=value):
                self.assert_cli_error(self.arguments() + ["--match-tolerance=" + value], "tolerance")
                self.assertFalse(self.output.exists())

    def test_nonempty_output_is_preserved_before_inputs_are_read(self):
        self.output.mkdir()
        sentinel = self.output / ".keep"
        sentinel.write_text("keep me", encoding="utf-8")
        # These are placeholders, deliberately not readable datasets.
        self.gt.touch()
        self.pred.touch()
        self.assert_cli_error(self.arguments(), "empty")
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_output_file_is_not_overwritten(self):
        self.output.write_text("keep me", encoding="utf-8")
        self.gt.touch()
        self.pred.touch()
        self.assert_cli_error(self.arguments(), "directory")
        self.assertEqual(self.output.read_text(encoding="utf-8"), "keep me")

    def test_output_symlink_is_not_followed(self):
        destination = self.root / "other-output"
        destination.mkdir()
        self.output.symlink_to(destination, target_is_directory=True)
        self.assert_cli_error(self.arguments(), "symlink")
        self.assertEqual(list(destination.iterdir()), [])

    def test_missing_input_crs_is_not_guessed(self):
        self.fixture(self.gt, crs=None)
        with self.assertRaisesRegex(ValueError, "CRS"):
            self.module.read_metric_gdf(self.gt, target_crs="EPSG:3857")

    def test_bad_prediction_crs_fails_before_gt_annotation_is_written(self):
        self.fixture(self.gt, with_nodes=False)
        self.fixture(self.pred, crs=None)
        self.assert_cli_error(self.arguments(), "CRS")
        self.assertFalse(self.output.exists())

    def test_reader_uses_runtime_crs_and_reprojects_in_meters(self):
        self.fixture(self.gt)
        gdf = self.module.read_metric_gdf(self.gt)
        self.assertEqual(gdf.crs.to_epsg(), 3857)
        self.assertAlmostEqual(gdf.geometry.iloc[0].length, 222.63898158654715, places=6)

    def test_main_refuses_nonempty_output_for_import_callers(self):
        self.output.mkdir()
        sentinel = self.output / "metrics_summary.csv"
        sentinel.write_text("existing results", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.module.main()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing results")

    def test_import_call_rejects_auxiliary_outputs_outside_protected_directory(self):
        self.fixture(self.gt, with_nodes=False)
        self.fixture(self.pred, with_nodes=False)
        for setting, destination in (
            ("GT_WITH_NODES_PATH", self.gt),
            ("PRED_WITH_NODES_PATH", self.pred),
            ("DEBUG_DIR", self.root / "existing-results"),
        ):
            with self.subTest(setting=setting), patch.object(self.module, setting, destination):
                # Fail before dataset reads/writes; no real overwrite is attempted.
                with patch.object(self.module, "read_metric_gdf", side_effect=AssertionError("Unsafe path reached data IO")):
                    with self.assertRaisesRegex(ValueError, "inside the output directory"):
                        self.module.main()
                self.assertFalse(self.output.exists())

    def test_import_call_rejects_colliding_auxiliary_outputs(self):
        self.fixture(self.gt)
        self.fixture(self.pred)
        with patch.object(self.module, "PRED_WITH_NODES_PATH", self.module.GT_WITH_NODES_PATH):
            with patch.object(self.module, "read_metric_gdf", side_effect=AssertionError("Colliding paths reached data IO")):
                with self.assertRaisesRegex(ValueError, "distinct"):
                    self.module.main()
        self.assertFalse(self.output.exists())

    def test_synthetic_cli_writes_metrics_and_annotations_in_requested_crs(self):
        self.fixture(self.gt, with_nodes=False)
        self.fixture(self.pred, with_nodes=False)
        self.output.mkdir()  # An existing empty directory is allowed.
        with ExitStack() as stack:
            # Skip only expensive rendering; graph construction, metrics and all
            # GIS/report writes run unchanged against the synthetic fixture.
            for name in ("plot_graph", "plot_matched_graph_comparison",
                         "plot_endpoint_matched_but_edge_unmatched",
                         "plot_unmatched_nodes_with_matched_neighbors"):
                stack.enter_context(patch.object(self.module, name))
            self.call_cli(self.arguments() + ["--match-tolerance", "2.5"])
        self.assertEqual(self.module.MATCH_TOLERANCE, 2.5)
        with (self.output / "metrics_summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row["region"] for row in rows},
                         {"overall", "junction", "lane_change", "road_segment"})
        overall = next(row for row in rows if row["region"] == "overall")
        for key in ("Topo-Precision", "Topo-Recall", "Topo-F1"):
            self.assertEqual(float(overall[key]), 1.0)
        for name in ("gt_with_nodes.shp", "pred_with_nodes.shp"):
            annotated = self.module.gpd.read_file(self.output / name)
            self.assertEqual(annotated.crs.to_epsg(), 3857)
            self.assertAlmostEqual(annotated.geometry.length.sum(), 222.63898158654715, places=6)
        self.assertTrue((self.output / "debug_reports/overall_edge_debug.txt").is_file())

    def test_documented_crs_and_default_tolerance_with_real_diagnostic_figures(self):
        for path in (self.gt, self.pred):
            self.fixture(path, crs="EPSG:32650",
                         coordinates=[(500000, 3300000), (500200, 3300000)])
        # Check real rendering at low resolution, not the production 600 DPI.
        with patch.object(self.module, "FIG_DPI", 30), \
                patch.object(self.module, "FIGSIZE_SQUARE", (3, 3)), \
                patch.object(self.module, "MATCH_TOLERANCE", 2.5):
            self.call_cli(self.arguments("EPSG:32650"))
            self.assertEqual(self.module.MATCH_TOLERANCE, 14.5)
        for suffix in ("png", "pdf", "svg"):
            self.assertGreater((self.output / ("gt_native_topology." + suffix)).stat().st_size, 0)
        self.assertTrue((self.output / "metrics_summary.csv").is_file())
        self.assertFalse((self.output / "gt_with_nodes.shp").exists())
        self.assertFalse((self.output / "pred_with_nodes.shp").exists())


if __name__ == "__main__":
    unittest.main()
