"""Shapefile validation only; no map geometry or attribute calculations."""
from pathlib import Path


def check_output(output, inputs=()):
    output = Path(output)
    if output.suffix.lower() != ".shp":
        raise ValueError("Output must have a .shp extension.")
    if output.resolve() in {Path(path).resolve() for path in inputs}:
        raise FileExistsError("Output must not overwrite an input.")
    if not output.parent.is_dir():
        raise FileNotFoundError("Output parent directory must already exist.")
    if any(output.parent.glob(output.stem + ".*")):
        raise FileExistsError("Output basename or sidecar already exists: " + str(output))
    return output


def read_crs(path, input_crs=None):
    from pyproj import CRS

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    projection = path.with_suffix(".prj")
    crs = CRS.from_wkt(projection.read_text()) if projection.is_file() else None
    supplied = CRS.from_user_input(input_crs) if input_crs else None
    if crs is not None and supplied is not None and crs != supplied:
        raise ValueError("Explicit input CRS disagrees with .prj: " + str(path))
    if crs is None and supplied is None:
        raise ValueError("Missing CRS: supply a .prj or explicit --input-crs: " + str(path))
    return crs if crs is not None else supplied


def write_crs(path, crs):
    Path(path).with_suffix(".prj").write_text(crs.to_wkt(version="WKT1_ESRI"))


def validate_attributes(reader, *, width=False):
    names = [field[0] for field in reader.fields[1:]]
    required = {"Label", "LaneNumber"}
    if width:
        required.update(("LaneWidth", "center", "type"))
    if len(names) < 7 or names[3].lower() != "id" or not required.issubset(names):
        raise ValueError("Original schema requires >=7 fields, id at index 3, and "
                         + ", ".join(sorted(required)))
    if width and (names[0] != "type" or names[2] != "center"):
        raise ValueError("Original width schema requires type at index 0 and center at index 2.")
    if reader.shapeType != 3 or any(len(shape.points) < 2 for shape in reader.shapes()):
        raise ValueError("Original attribute calculations require 2D polylines with >=2 points.")


def attribute_parser(description):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, help="Input shapefile; field order is preserved.")
    parser.add_argument("--output", required=True, help="New output shapefile; no overwrites.")
    parser.add_argument("--input-crs", help="Explicit CRS only when .prj is absent; never relabels an existing CRS.")
    return parser
