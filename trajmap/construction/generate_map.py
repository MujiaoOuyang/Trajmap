#!/usr/bin/env python3
"""Headless trajectory-guided lane and optional arrow-map generation.

This entry preserves the core's lane-transition exporter and uses the recovered
arrow module. Historical junction-graph construction is not supplied here.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import traceback


SOURCE_DIR = Path(__file__).resolve().parent


def load_module(name):
    spec = importlib.util.spec_from_file_location("_trajmap_" + name, SOURCE_DIR / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_core():
    import matplotlib
    matplotlib.use("Agg")
    core = load_module("process_lane_segments")
    for name in dir(core):
        if name.startswith("visualize_") and callable(getattr(core, name)):
            setattr(core, name, lambda *args, **kwargs: None)
    return core


def coordinate_functions(input_crs, enu_origin):
    import numpy as np
    from pyproj import CRS, Geod, Transformer

    if (input_crs is None) == (enu_origin is None):
        raise ValueError("Specify exactly one of --input-crs and --enu-origin.")
    if input_crs:
        crs = CRS.from_user_input(input_crs)
        if not crs.is_projected or not crs.axis_info or any(
            not math.isclose(axis.unit_conversion_factor, 1.0) for axis in crs.axis_info[:2]
        ):
            raise ValueError("Input CRS must be projected with meter units.")
        return (lambda point: np.asarray(point, dtype=float)), crs, (
            lambda a, b: float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2])))

    lat, lon, height = map(float, enu_origin)
    if not all(math.isfinite(v) for v in (lat, lon, height)) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Invalid ENU origin: expected latitude, longitude, height.")
    to_ecef = Transformer.from_crs(4979, 4978, always_xy=True)
    to_wgs = Transformer.from_crs(4978, 4979, always_xy=True)
    origin = np.array(to_ecef.transform(lon, lat, height))
    phi, lam = math.radians(lat), math.radians(lon)
    rotation = np.array([
        [-math.sin(lam), -math.sin(phi) * math.cos(lam), math.cos(phi) * math.cos(lam)],
        [math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi) * math.sin(lam)],
        [0.0, math.cos(phi), math.sin(phi)],
    ])

    def convert(point):
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("ENU points must contain finite east, north, up coordinates.")
        return to_wgs.transform(*(origin + rotation @ point))

    geod = Geod(ellps="WGS84")
    return convert, CRS.from_epsg(4326), lambda a, b: abs(geod.inv(a[0], a[1], b[0], b[1])[2])


def prepare_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def discover_segments(directory, selected_ids=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("Segment directory does not exist: " + str(directory))
    pairs = []
    for cloud in sorted(directory.glob("lane_segment_*.pcd")):
        if not re.fullmatch(r"lane_segment_(?:incomplete_)?\d+", cloud.stem):
            continue
        if selected_ids is not None and int(cloud.stem.split("_")[-1]) not in selected_ids:
            continue
        poses = cloud.with_name(cloud.stem + "_poses.txt")
        if not poses.is_file():
            raise ValueError("Missing trajectory for " + str(cloud))
        pairs.append((cloud, poses))
    if not pairs:
        raise ValueError("No paired lane_segment_*.pcd / *_poses.txt files found.")
    return pairs


def validate_poses(path):
    import numpy as np
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "#" in stripped:
                raise ValueError("Inline TUM comments are unsupported by the original pose readers.")
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape[0] < 2 or rows.shape[1] != 8 or not np.isfinite(rows).all():
        raise ValueError("TUM trajectory must contain >=2 finite rows of 8 values.")
    if np.any(np.linalg.norm(rows[:, 4:8], axis=1) == 0):
        raise ValueError("Zero quaternion in trajectory.")
    if np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing.")
    return rows


def publish_map(frame, output, filename):
    import geopandas as gpd
    output = Path(output)
    with tempfile.TemporaryDirectory(prefix="export_", dir=output) as staging:
        target = Path(staging) / filename
        frame.to_file(target, encoding="UTF-8")
        check = gpd.read_file(target)
        if len(check) != len(frame) or check.crs != frame.crs or not check.geometry.is_valid.all():
            raise ValueError("Shapefile read-back verification failed.")
        # Publish the .shp last, after its complete, verified sidecar bundle.
        files = sorted(Path(staging).iterdir(), key=lambda p: p.suffix == ".shp")
        if any((output / path.name).exists() for path in files):
            raise FileExistsError("Refusing to replace an existing Shapefile bundle.")
        for path in files:
            path.rename(output / path.name)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}


def arrow_frame(arrows, segment_id, convert, crs, start_id=0):
    """Export closed XY box boundaries and independently transformed cluster means."""
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon

    records = []
    for index, arrow in enumerate(arrows):
        vertices = np.asarray(arrow["bbox_vertices"], dtype=float)
        center = np.asarray(arrow["center"], dtype=float)
        if vertices.shape != (4, 3) or center.shape != (3,) or not (
            np.isfinite(vertices).all() and np.isfinite(center).all()
        ):
            raise ValueError("Arrow vertices/cluster center must be finite XYZ coordinates.")
        transformed = np.asarray([convert(point) for point in vertices])
        cluster = np.asarray(convert(center))
        if not np.isfinite(transformed).all() or not np.isfinite(cluster).all():
            raise ValueError("Arrow coordinate transformation returned nonfinite values.")
        polygon = Polygon(transformed[:, :2])
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError("Arrow box must be a valid nonzero-area polygon.")
        records.append(dict(
            id=start_id + index, type="arrow", lane_type="", segment_id=segment_id,
            left_lane=-1, right_lane=-1, arrow_id=int(arrow["label"]), group_id=int(arrow["group_id"]),
            cluster_x=cluster[0], cluster_y=cluster[1], cluster_z=cluster[2],
            geometry=LineString(polygon.exterior.coords)))
    columns = ["id", "type", "lane_type", "segment_id", "left_lane", "right_lane",
               "arrow_id", "group_id", "cluster_x", "cluster_y", "cluster_z", "geometry"]
    return gpd.GeoDataFrame(records, columns=columns, geometry="geometry", crs=crs)


def run(args):
    import numpy as np
    import geopandas as gpd
    import pandas as pd

    convert, crs, distance = coordinate_functions(args.input_crs, args.enu_origin)
    if args.min_displacement <= 0 or not math.isfinite(args.min_displacement):
        raise ValueError("--min-displacement must be positive and finite.")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be between 0 and 2**32 - 1.")
    if args.segment_dir:
        if args.poses:
            raise ValueError("--poses is only used with --pcd; segmented inputs have paired pose files.")
        pairs = discover_segments(args.segment_dir, args.segment_ids)
        if args.segment_ids is not None:
            requested = set(args.segment_ids)
            found = {int(pair[0].stem.split("_")[-1]) for pair in pairs}
            if requested != found:
                raise ValueError("Requested segment IDs not found: " + str(sorted(requested - found)))
    else:
        if not args.poses or not Path(args.pcd).is_file() or not Path(args.poses).is_file():
            raise ValueError("--pcd requires an existing point cloud and --poses TUM trajectory.")
        if args.segment_ids:
            raise ValueError("--segment-ids requires --segment-dir.")
        validate_poses(args.poses)

    output = prepare_output(args.output_dir)
    manifest = {
        "status": "running", "scope": "lane_and_arrow_geometry" if args.with_arrows else "lane_geometry_with_core_transition_export",
        "missing_stages": ["historical_junction_graph_export", "lane_width_export"],
        "disabled_stages": [] if args.with_arrows else ["arrow_rectangle_export"],
        "parameters": vars(args), "output_crs": crs.to_string(),
        "coordinate_conversion": "projected_identity" if args.input_crs else "ENU_to_ECEF_to_WGS84",
        "segments": [],
    }
    frames = []
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
        np.random.seed(args.seed)
        source_names = ["generate_map.py", "process_lane_segments.py", "save_stopline_points.py"]
        if args.with_arrows:
            source_names.append("arrow_generation.py")
        manifest["source_files"] = [fingerprint(SOURCE_DIR / name) for name in source_names]
        with (output / "generation.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            core = load_core()
            arrows = load_module("arrow_generation") if args.with_arrows else None
            if not args.segment_dir:
                manifest["inputs"] = [fingerprint(args.pcd), fingerprint(args.poses)]
                splitter = load_module("save_stopline_points")
                splitter.main(["--input_file", str(Path(args.pcd).resolve()),
                               "--pose_file", str(Path(args.poses).resolve()),
                               "--output_dir", str(output / "segments")])
                pairs = discover_segments(output / "segments")
            for index, (cloud, poses) in enumerate(pairs, 1):
                record = {"segment_id": index, "input": fingerprint(cloud),
                          "poses": fingerprint(poses), "status": "pending"}
                manifest["segments"].append(record)
                try:
                    arrow_data = []
                    arrow_calls = []

                    def collect_arrows(*callback_args, **callback_kwargs):
                        # Diagnostics in the arrow stage must not change later lane fits.
                        state = np.random.get_state()
                        try:
                            arrow_data.extend(arrows.deal_arrow_process(*callback_args, **callback_kwargs))
                            arrow_calls.append(True)
                        finally:
                            np.random.set_state(state)

                    # Normalize incomplete-tail filenames without touching source data.
                    with tempfile.TemporaryDirectory(prefix="segment_", dir=output) as staging:
                        shutil.copyfile(cloud, Path(staging) / "lane_segment_001.pcd")
                        shutil.copyfile(poses, Path(staging) / "lane_segment_001_poses.txt")
                        validate_poses(poses)
                        lanes, generated_type, count = core.process_lane_segment(
                            staging, 1, min_displacement=args.min_displacement,
                            visualize=False, visualize_3d=False,
                            **({"arrow_processor": collect_arrows} if args.with_arrows else {}))
                    if not lanes:
                        raise ValueError("Core produced no lanes; see log for thresholds/turn-splitting requirements.")
                    frame = core.save_segment_shapefile_with_numbers(
                        lanes, generated_type, index, count, None,
                        point_transform=convert, output_crs=crs,
                        distance_function=distance, return_frame=True)
                    if frame.empty or not frame.geometry.is_valid.all() or any(g.is_empty or g.length <= 0 for g in frame.geometry):
                        raise ValueError("Exporter returned empty or invalid lane geometry.")
                    frame["lane_type"] = frame["type"].map(
                        {"left solid": "solid", "right solid": "solid", "solid": "solid", "dashed": "dashed"})
                    if frame["lane_type"].isna().any():
                        raise ValueError("Unexpected lane subtype returned by core.")
                    frame["type"] = "lane"
                    lane_features = len(frame)
                    if args.with_arrows:
                        if len(arrow_calls) != 1:
                            raise RuntimeError("Lane core did not execute exactly one arrow stage.")
                        if arrow_data:
                            boxes = arrow_frame(arrow_data, index, convert, crs,
                                                start_id=int(frame["id"].max()) + 1)
                            frame = gpd.GeoDataFrame(pd.concat([frame, boxes], ignore_index=True), crs=crs)
                    frames.append(frame)
                    record.update(status="ok", generated_type=int(generated_type), features=len(frame),
                                  lane_features=lane_features, arrow_features=len(arrow_data))
                except Exception as error:
                    record.update(status="failed", error=str(error))
                    print("FAILED", cloud, repr(error))
                    traceback.print_exc(file=log)
        failed = any(item["status"] != "ok" for item in manifest["segments"])
        if not frames or (failed and not args.allow_partial):
            raise RuntimeError("Map generation incomplete. Inspect generation.log/run_manifest.json; no map was published.")
        result = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs)
        result["edge_id"] = np.arange(len(result))
        stem = "vector_map" if args.with_arrows else "lane_map"
        filename = stem + ("_partial.shp" if failed else ".shp")
        publish_map(result, output, filename)
        manifest.update(status="partial" if failed else "success", features=len(result), output=filename,
                        lane_features=int((result["type"] == "lane").sum()),
                        arrow_features=int((result["type"] == "arrow").sum()))
        print(str(output / filename))
        return 2 if failed else 0
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--segment-dir", help="Paired semantic lane_segment_*.pcd and *_poses.txt")
    inputs.add_argument("--pcd", help="Global RGB semantic point cloud; requires --poses")
    parser.add_argument("--poses", help="TUM: timestamp x y z qx qy qz qw; same frame as cloud")
    frame = parser.add_mutually_exclusive_group(required=True)
    frame.add_argument("--input-crs", help="Projected, meter-based CRS already used by cloud/trajectory")
    frame.add_argument("--enu-origin", type=float, nargs=3, metavar=("LAT", "LON", "HEIGHT"))
    parser.add_argument("--output-dir", required=True, help="New directory; existing paths are refused")
    parser.add_argument("--segment-ids", nargs="+", type=int)
    parser.add_argument("--min-displacement", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-arrows", action="store_true",
                        help="Generate arrow rectangles from semantic points and export vector_map.shp")
    parser.add_argument("--allow-partial", action="store_true", help="Save explicitly named partial map, still exit 2")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        print("ERROR:", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
