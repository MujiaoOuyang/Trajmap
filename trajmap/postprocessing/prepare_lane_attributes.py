"""Adapt construction lane exports ONLY for cal_shp_lane_info (width variant).

Requires nonempty, valid, finite, single-part 2D lane polylines, fields type,
lane_type and id, and a declared geographic CRS or EPSG:32650. No geometry is
split, fitted, reprojected or otherwise changed. The width method still uses
only the first two vertices; these must be distinct. This adapter does not
change that convention.

The first seven fields become type, lane_type, center, Id, source, target,
Label. Id is a new unique integer for the legacy scan, NOT a global graph/node
identifier. Original id becomes orig_id; edge_id, segment_id and other input
attributes are retained. Missing source/target/center remain unknown, not
inferred. --angle-threshold is mandatory: 10 writes numeric Label=0; 60 writes
numeric Label=1. This is a caller-selected angle gate, not an inferred lane
label. LaneNumber and LaneWidth start null and are calculated by the width tool.
Existing Label/LaneNumber/LaneWidth values are retained as orig_label/orig_lnum/
orig_width. Conflicting preservation names are rejected, never overwritten.

Do NOT pass this output directly to cal_shp_lane_info_lanenumber_label: that
separate variant interprets target at index 5 as a numeric skip gate, not as a
node identifier. No topology inference or field-value coercion is performed.
The width tool overwrites Label with its result; do not infer a new angle gate
from that result on a subsequent run. Outputs and existing sidecars are protected.
"""
import argparse
import math

if __package__:
    from ._shapefile_io import check_output, read_crs, write_crs
else:
    from _shapefile_io import check_output, read_crs, write_crs


def prepare_lane_attributes(input_shp, output_shp, *, angle_threshold):
    """Write an explicit legacy width-input schema and return the output path."""
    if angle_threshold not in (10, 60):
        raise ValueError("angle_threshold must be explicitly selected as 10 or 60.")
    output = check_output(output_shp, (input_shp,))
    crs = read_crs(input_shp)
    from pyproj import CRS
    import shapefile
    from shapely.geometry import LineString

    if not crs.is_geographic and crs != CRS.from_epsg(32650):
        raise ValueError("Width input CRS must be geographic or EPSG:32650.")

    with shapefile.Reader(str(input_shp)) as reader:
        fields = reader.fields[1:]
        names = [field[0] for field in fields]
        if len({name.lower() for name in names}) != len(names):
            raise ValueError("Input field-name collision (case insensitive).")
        if not {"type", "lane_type", "id"}.issubset(names):
            raise ValueError("Construction schema requires type, lane_type and id fields.")
        if reader.shapeType != shapefile.POLYLINE:
            raise ValueError("Only single-part 2D lane polylines are supported.")
        records = reader.shapeRecords()
        if not records:
            raise ValueError("Input lane map must not be empty.")
        for record in records:
            if record.record["type"] != "lane":
                raise ValueError("Only pure type=lane inputs are supported; arrows are not accepted.")
            geometry = record.shape
            if (geometry.shapeType != shapefile.POLYLINE or len(geometry.parts) != 1
                    or len(geometry.points) < 2):
                raise ValueError("Every feature must be a nonempty single-part 2D polyline.")
            if any(len(point) != 2 or not all(math.isfinite(value) for value in point)
                   for point in geometry.points):
                raise ValueError("Lane coordinates must be finite and 2D.")
            if tuple(geometry.points[0]) == tuple(geometry.points[1]):
                raise ValueError("The first two vertices must be distinct for the width method.")
            line = LineString(geometry.points)
            if line.is_empty or not line.is_valid or line.length <= 0:
                raise ValueError("Every lane polyline must be nonempty and valid with positive length.")

        definitions = {field[0]: field for field in fields}
        output_fields = [
            definitions["type"], definitions["lane_type"],
            definitions.get("center", ("center", "C", 254, 0)),
            ("Id", "N", 18, 0),
            definitions.get("source", ("source", "N", 18, 0)),
            definitions.get("target", ("target", "N", 18, 0)),
            ("Label", "N", 10, 0), ("LaneNumber", "N", 18, 0),
            ("LaneWidth", "F", 20, 6),
        ]
        preserved_names = {"id": "orig_id", "Label": "orig_label",
                           "LaneNumber": "orig_lnum", "LaneWidth": "orig_width"}
        existing = {field[0].lower() for field in output_fields}
        extras = []
        for field in fields:
            name = field[0]
            if name in ("type", "lane_type", "center", "source", "target"):
                continue
            preserved = preserved_names.get(name, name)
            if preserved.lower() in existing:
                raise ValueError("Preserved field-name collision: " + preserved)
            existing.add(preserved.lower())
            output_fields.append((preserved, *field[1:]))
            extras.append(name)

        with shapefile.Writer(str(output), shapeType=shapefile.POLYLINE) as writer:
            for field in output_fields:
                writer.field(*field)
            for scan_id, record in enumerate(records):
                values = record.record.as_dict()
                writer.record(values["type"], values["lane_type"], values.get("center"),
                              scan_id, values.get("source"), values.get("target"),
                              0 if angle_threshold == 10 else 1, None, None,
                              *(values[name] for name in extras))
                writer.shape(record.shape)
        write_crs(output, crs)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Construction lane-only Shapefile with .prj.")
    parser.add_argument("--output", required=True, help="New Shapefile basename; no overwrites.")
    parser.add_argument("--angle-threshold", required=True, type=int, choices=(10, 60),
                        help="Explicit angle gate for the original width calculation, in degrees.")
    args = parser.parse_args(argv)
    print(prepare_lane_attributes(args.input, args.output, angle_threshold=args.angle_threshold))


if __name__ == "__main__":
    main()
