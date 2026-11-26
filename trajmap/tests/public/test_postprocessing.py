"""Packaging checks; synthetic files live only in temporary directories."""
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString


ROOT = Path(__file__).resolve().parents[2]
MODULES = ("fusion_multishp", "cal_shp_lane_info",
           "cal_shp_lane_info_lanenumber_label", "convert_3d_to_2d",
           "prepare_lane_attributes")
HAS_PYSHP = importlib.util.find_spec("shapefile") is not None


class PackagingTests(unittest.TestCase):
    def test_help_and_import_without_optional_dependencies_or_io(self):
        for name in MODULES:
            with self.subTest(module=name), tempfile.TemporaryDirectory() as directory:
                script = ROOT / "postprocessing" / (name + ".py")
                self.assertTrue(script.is_file(), str(script))
                result = subprocess.run([sys.executable, "-B", "-S", str(script), "--help"],
                                        cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                code = (
                    "import os, sys; before = dict(os.environ); "
                    "sys.path.insert(0, " + repr(str(ROOT)) + "); "
                    "import postprocessing." + name + "; "
                    "assert dict(os.environ) == before"
                )
                result = subprocess.run([sys.executable, "-B", "-S", "-c", code],
                                        cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertEqual(os.listdir(directory), [])


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "postprocessing/fusion_multishp.py").is_file())
        self.fusion = importlib.import_module("postprocessing.fusion_multishp")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def source(self, name, *, crs="EPSG:32650", identifier="id", kind="lane"):
        path = self.directory / (name + ".shp")
        frame = gpd.GeoDataFrame({"type": [kind], identifier: [17], "extra": ["drop"]},
                                geometry=[LineString([(500000, 3500000), (500010, 3500000)])],
                                crs=crs)
        frame.to_file(path, index=False)
        return path

    @unittest.skipUnless(HAS_PYSHP, "Strict merge requires PyShp")
    def test_strict_merge_preserves_schema_ids_coordinates_and_crs(self):
        first, second = self.source("first"), self.source("second", identifier="Id")
        output = self.directory / "merged.shp"
        self.fusion.twofileFusion_strict(first, second, output)
        frame = gpd.read_file(output)
        self.assertEqual(list(frame.columns), ["type", "lane_type", "center", "Id",
                         "source", "target", "Label", "LaneNumber", "LaneWidth", "RAppend", "geometry"])
        self.assertEqual(frame.Id.tolist(), [17, 17])
        self.assertEqual(frame.crs, CRS.from_epsg(32650))
        self.assertEqual(list(frame.geometry.iloc[0].coords), [(500000, 3500000), (500010, 3500000)])

    @unittest.skipUnless(HAS_PYSHP, "Strict merge requires PyShp")
    def test_merge_rejects_missing_mismatched_crs_and_existing_sidecars(self):
        first = self.source("first")
        second = self.source("second", crs="EPSG:4326")
        output = self.directory / "merged.shp"
        with self.assertRaisesRegex(ValueError, "CRS"):
            self.fusion.twofileFusion_strict(first, second, output)
        self.assertFalse(output.exists())
        second.with_suffix(".prj").unlink()
        with self.assertRaisesRegex(ValueError, "CRS"):
            self.fusion.twofileFusion_strict(first, second, output)
        output.with_suffix(".dbf").write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self.fusion.twofileFusion_strict(first, first, output)
        self.assertEqual(output.with_suffix(".dbf").read_bytes(), b"keep")

    def test_arrows_normalize_type_and_id_without_changing_geometry(self):
        base = self.source("base", identifier="Id")
        folder = self.directory / "arrows"
        folder.mkdir()
        arrow = gpd.GeoDataFrame({"type": ["Arrow", "lane"], "id": [9, 99]},
                                geometry=[LineString([(1, 2), (3, 4)])] * 2, crs="EPSG:32650")
        arrow.to_file(folder / "section.shp", index=False)
        output = self.directory / "with_arrows.shp"
        self.fusion.extract_and_merge_arrows(base, folder, output)
        frame = gpd.read_file(output)
        self.assertEqual(frame["type"].tolist(), ["lane", "arrow"])
        self.assertEqual(frame["id"].tolist(), [17, 9])
        self.assertNotIn("Id", frame.columns)
        self.assertNotIn("extra", frame.columns)
        self.assertEqual(list(frame.geometry.iloc[1].coords), [(1, 2), (3, 4)])
        self.assertEqual(frame.crs, CRS.from_epsg(32650))
        with self.assertRaises(FileExistsError):
            self.fusion.extract_and_merge_arrows(base, folder, output)

    def test_arrows_reject_crs_mismatch_before_writing(self):
        base = self.source("base")
        folder = self.directory / "arrows"
        folder.mkdir()
        frame = gpd.GeoDataFrame({"type": ["arrow"]},
                                geometry=[LineString([(1, 2), (3, 4)])], crs="EPSG:4326")
        frame.to_file(folder / "section.shp", index=False)
        output = self.directory / "bad.shp"
        with self.assertRaisesRegex(ValueError, "CRS"):
            self.fusion.extract_and_merge_arrows(base, folder, output)
        self.assertFalse(output.exists())


@unittest.skipUnless(HAS_PYSHP, "Attribute calculations require PyShp")
class AttributeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.input = self.directory / "lanes.shp"
        self.frame = gpd.GeoDataFrame({
            "type": ["lane"] * 3, "lane_type": ["dashed"] * 3,
            "center": ["1,2"] * 3, "Id": [1, 2, 3], "source": [0] * 3,
            "target": [0] * 3, "Label": [0] * 3,
            "LaneNumber": [0] * 3, "LaneWidth": [0.] * 3,
        }, geometry=[LineString([(500000, 3500000 + y), (500010, 3500000 + y)])
                     for y in (0, 3, 6)], crs="EPSG:32650")
        self.frame.to_file(self.input, index=False)

    def test_width_preserves_original_calculations_and_field_order(self):
        module = importlib.import_module("postprocessing.cal_shp_lane_info")
        output = self.directory / "width.shp"
        module.process(self.input, output)
        frame = gpd.read_file(output)
        self.assertEqual(list(frame.columns), list(self.frame.columns))
        self.assertEqual(frame.LaneNumber.tolist(), [2, 2, 0])
        self.assertEqual(frame.Label.tolist(), [1, 2, -1])
        self.assertEqual(frame.LaneWidth.tolist(), [3., 3., 0.])
        self.assertEqual(frame.center.tolist(), ["1,2"] * 3)
        self.assertEqual(frame.crs, self.frame.crs)
        self.assertTrue(frame.geometry.equals(self.frame.geometry))
        with self.assertRaises(FileExistsError):
            module.process(self.input, output)

    def test_vertical_parallel_lines_keep_original_width_and_count_semantics(self):
        module = importlib.import_module("postprocessing.cal_shp_lane_info")
        self.frame.geometry = [LineString([(500000 - y, 3500000), (500000 - y, 3500010)])
                               for y in (0, 3, 6)]
        vertical = self.directory / "vertical.shp"
        self.frame.to_file(vertical, index=False)
        output = self.directory / "vertical_width.shp"
        module.process(vertical, output)
        frame = gpd.read_file(output)
        self.assertEqual(frame.LaneNumber.tolist(), [2, 2, 0])
        self.assertEqual(frame.Label.tolist(), [1, 2, -1])
        self.assertEqual(frame.LaneWidth.tolist(), [3., 3., 0.])
        self.assertTrue(frame.geometry.equals(self.frame.geometry))

    @unittest.skipUnless(importlib.util.find_spec("rtree"), "Label variant requires Rtree")
    def test_label_variant_preserves_original_counts(self):
        module = importlib.import_module("postprocessing.cal_shp_lane_info_lanenumber_label")
        output = self.directory / "labels.shp"
        module.process(self.input, output)
        frame = gpd.read_file(output)
        self.assertEqual(frame.LaneNumber.tolist(), [3, 3, 3])
        self.assertEqual(frame.Label.tolist(), [1, 2, 3])
        self.assertTrue(frame.geometry.equals(self.frame.geometry))
        self.assertEqual(list(frame.columns), list(self.frame.columns))

    def test_schema_and_unsupported_projection_fail_before_writing(self):
        module = importlib.import_module("postprocessing.cal_shp_lane_info")
        output = self.directory / "invalid.shp"
        malformed = self.directory / "malformed.shp"
        self.frame.drop(columns=["LaneWidth"]).to_file(malformed, index=False)
        with self.assertRaisesRegex(ValueError, "schema"):
            module.process(malformed, output)
        self.assertFalse(output.exists())
        different = self.directory / "different.shp"
        self.frame.to_crs(32651).to_file(different, index=False)
        with self.assertRaisesRegex(ValueError, "32650"):
            module.process(different, output)
        self.assertFalse(output.exists())

    def test_missing_projection_requires_explicit_crs(self):
        module = importlib.import_module("postprocessing.cal_shp_lane_info")
        self.input.with_suffix(".prj").unlink()
        output = self.directory / "explicit.shp"
        with self.assertRaisesRegex(ValueError, "Missing CRS"):
            module.process(self.input, output)
        module.process(self.input, output, input_crs="EPSG:32650")
        self.assertEqual(gpd.read_file(output).crs, CRS.from_epsg(32650))


@unittest.skipUnless(HAS_PYSHP, "Lane attribute preparation requires PyShp")
class PrepareAttributeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "postprocessing/prepare_lane_attributes.py").is_file())
        self.module = importlib.import_module("postprocessing.prepare_lane_attributes")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.frame = gpd.GeoDataFrame({
            "type": ["lane"] * 3, "lane_type": ["solid", "dashed", "solid"],
            "id": [7, 7, 8], "edge_id": [10, 11, 12], "segment_id": [1, 2, 2],
            "left_lane": [1, 2, 3], "right_lane": [0, 1, 2],
        }, geometry=[LineString([(500000, 3500000 + y), (500010, 3500000 + y)])
                     for y in (0, 3, 6)], crs="EPSG:32650")

    def save_input(self, frame=None, name="input"):
        path = self.directory / (name + ".shp")
        (self.frame if frame is None else frame).to_file(path, index=False)
        return path

    def test_prepare_and_width_chain_preserves_metadata_for_both_orientations(self):
        import shapefile
        width = importlib.import_module("postprocessing.cal_shp_lane_info")
        for vertical in (False, True):
            for threshold, gate in ((10, 0), (60, 1)):
                with self.subTest(vertical=vertical, threshold=threshold):
                    source_frame = self.frame.copy()
                    if vertical:
                        source_frame.geometry = [
                            LineString([(500000 - y, 3500000), (500000 - y, 3500010)])
                            for y in (0, 3, 6)]
                    tag = str(vertical) + str(threshold)
                    source = self.save_input(source_frame, "input" + tag)
                    original = {p.name: p.read_bytes() for p in self.directory.glob(source.stem + ".*")}
                    prepared = self.directory / ("prepared" + tag + ".shp")
                    self.module.prepare_lane_attributes(source, prepared, angle_threshold=threshold)
                    with shapefile.Reader(str(prepared)) as reader:
                        self.assertEqual([f[0] for f in reader.fields[1:8]],
                                         ["type", "lane_type", "center", "Id", "source", "target", "Label"])
                        self.assertEqual(reader.fields[7][1], "N")
                        self.assertEqual([r[6] for r in reader.records()], [gate] * 3)
                        for record in reader.records():
                            values = record.as_dict()
                            self.assertIsNone(values["source"])
                            self.assertIsNone(values["target"])
                            self.assertIsNone(values["LaneNumber"])
                            self.assertIsNone(values["LaneWidth"])
                    output = self.directory / ("width" + tag + ".shp")
                    width.process(prepared, output)
                    result = gpd.read_file(output)
                    self.assertTrue(result["Id"].is_unique)
                    for column in ("edge_id", "segment_id", "left_lane", "right_lane", "lane_type"):
                        self.assertEqual(result[column].tolist(), source_frame[column].tolist())
                    self.assertEqual(result.orig_id.tolist(), source_frame.id.tolist())
                    self.assertEqual(result.LaneNumber.tolist(), [2, 2, 0])
                    self.assertEqual(result.Label.tolist(), [1, 2, -1])
                    self.assertEqual(result.LaneWidth.tolist(), [3., 3., 0.])
                    self.assertTrue(result.source.isna().all() and result.target.isna().all())
                    self.assertTrue(result.geometry.equals(source_frame.geometry))
                    self.assertEqual(original, {p.name: p.read_bytes()
                                               for p in self.directory.glob(source.stem + ".*")})

    def test_existing_attributes_are_preserved_without_inferring_angle_gate(self):
        self.frame["source"] = [101, 102, 103]
        self.frame["target"] = [201, 202, 203]
        self.frame["center"] = ["original"] * 3
        self.frame["Label"] = [99] * 3
        self.frame["LaneNumber"] = [8] * 3
        self.frame["LaneWidth"] = [4.5] * 3
        source = self.save_input()
        output = self.directory / "prepared.shp"
        self.module.prepare_lane_attributes(source, output, angle_threshold=10)
        result = gpd.read_file(output)
        self.assertEqual(result.Label.tolist(), [0] * 3)
        self.assertEqual(result.orig_label.tolist(), [99] * 3)
        self.assertEqual(result.orig_lnum.tolist(), [8] * 3)
        self.assertEqual(result.orig_width.tolist(), [4.5] * 3)
        self.assertEqual(result.source.tolist(), [101, 102, 103])
        self.assertEqual(result.target.tolist(), [201, 202, 203])
        self.assertEqual(result.center.tolist(), ["original"] * 3)
        self.assertTrue(result.LaneNumber.isna().all() and result.LaneWidth.isna().all())

    def test_rejects_arrow_multipart_3d_invalid_and_unsupported_crs(self):
        cases = {}
        cases["arrow"] = self.frame.assign(type="arrow")
        multipart = self.frame.iloc[:1].copy()
        multipart.geometry = [MultiLineString([[(0, 0), (1, 0)], [(2, 0), (3, 0)]])]
        cases["multipart"] = multipart
        three = self.frame.iloc[:1].copy()
        three.geometry = [LineString([(0, 0, 1), (1, 0, 2)])]
        cases["three"] = three
        invalid = self.frame.iloc[:1].copy()
        invalid.geometry = [LineString([(1, 1), (1, 1)])]
        cases["invalid"] = invalid
        cases["projection"] = self.frame.to_crs(32651)
        for name, frame in cases.items():
            with self.subTest(case=name):
                source = self.save_input(frame, name)
                output = self.directory / (name + "_out.shp")
                with self.assertRaises(ValueError):
                    self.module.prepare_lane_attributes(source, output, angle_threshold=10)
                self.assertEqual(list(self.directory.glob(output.stem + ".*")), [])

    def test_repeated_first_vertices_are_rejected_before_writing(self):
        frame = self.frame.iloc[:1].copy()
        frame.geometry = [LineString([(500000, 3500000), (500000, 3500000),
                                      (500010, 3500000)])]
        self.assertTrue(frame.geometry.iloc[0].is_valid)
        self.assertGreater(frame.geometry.iloc[0].length, 0)
        source = self.save_input(frame)
        output = self.directory / "repeated_out.shp"
        with self.assertRaisesRegex(ValueError, "first two vertices"):
            self.module.prepare_lane_attributes(source, output, angle_threshold=10)
        self.assertEqual(list(self.directory.glob(output.stem + ".*")), [])

    def test_declared_geographic_crs_is_preserved_and_missing_crs_rejected(self):
        source = self.save_input(self.frame.to_crs(4326))
        output = self.directory / "geographic.shp"
        self.module.prepare_lane_attributes(source, output, angle_threshold=60)
        self.assertEqual(gpd.read_file(output).crs, CRS.from_epsg(4326))
        source.with_suffix(".prj").unlink()
        with self.assertRaisesRegex(ValueError, "CRS"):
            self.module.prepare_lane_attributes(source, self.directory / "missing.shp", angle_threshold=10)

    def test_output_sidecars_and_original_id_collisions_are_protected(self):
        source = self.save_input()
        output = self.directory / "exists.shp"
        output.with_suffix(".dbf").write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self.module.prepare_lane_attributes(source, output, angle_threshold=10)
        self.assertEqual(output.with_suffix(".dbf").read_bytes(), b"keep")
        with self.assertRaises(FileExistsError):
            self.module.prepare_lane_attributes(source, source, angle_threshold=10)
        self.frame["orig_id"] = [123] * 3
        collision = self.save_input(name="collision")
        with self.assertRaisesRegex(ValueError, "collision"):
            self.module.prepare_lane_attributes(collision, self.directory / "bad.shp", angle_threshold=10)

    def test_empty_input_is_rejected_without_output(self):
        import shapefile
        source = self.directory / "empty.shp"
        with shapefile.Writer(str(source), shapeType=shapefile.POLYLINE) as writer:
            writer.field("type", "C")
            writer.field("lane_type", "C")
            writer.field("id", "N")
        source.with_suffix(".prj").write_text(CRS.from_epsg(32650).to_wkt(version="WKT1_ESRI"))
        output = self.directory / "empty_out.shp"
        with self.assertRaisesRegex(ValueError, "empty"):
            self.module.prepare_lane_attributes(source, output, angle_threshold=10)
        self.assertEqual(list(self.directory.glob(output.stem + ".*")), [])

    def test_cli_requires_explicit_10_or_60_and_runs(self):
        source = self.save_input()
        output = self.directory / "cli.shp"
        command = [sys.executable, "-B", str(ROOT / "postprocessing/prepare_lane_attributes.py"),
                   "--input", str(source), "--output", str(output)]
        for arguments in ([], ["--angle-threshold", "30"]):
            result = subprocess.run(command + arguments, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            self.module.prepare_lane_attributes(source, output, angle_threshold=30)
        result = subprocess.run(command + ["--angle-threshold", "60"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(gpd.read_file(output).Label.tolist(), [1] * 3)


class ConversionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("osgeo"), "3D conversion requires GDAL Python bindings")
    def test_conversion_preserves_xy_and_records_endpoint_z(self):
        module = importlib.import_module("postprocessing.convert_3d_to_2d")
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "three.shp"
            output = Path(directory) / "two.shp"
            gpd.GeoDataFrame({"id": [7], "center": ["old"]},
                             geometry=[LineString([(1, 2, 3), (4, 5, 6)])],
                             crs="EPSG:32650").to_file(input_path, index=False)
            module.convert_3d_to_2d(str(input_path), str(output))
            frame = gpd.read_file(output)
            self.assertEqual(frame.center.tolist(), ["3,6"])
            self.assertEqual(list(frame.geometry.iloc[0].coords), [(1, 2), (4, 5)])
            self.assertEqual(frame.crs, CRS.from_epsg(32650))
            with self.assertRaises(FileExistsError):
                module.convert_3d_to_2d(str(input_path), str(output))


if __name__ == "__main__":
    unittest.main()
