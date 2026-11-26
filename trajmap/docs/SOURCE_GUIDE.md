# Source Guide

## Overview

The code is organized into semantic point-cloud preparation, lane-line
reconstruction, Shapefile postprocessing and map evaluation. Each stage has
its own inputs and entry points. This guide describes the main functions,
coordinate conventions and supported workflows.

## Runnable Lane Generation

Use [`construction/generate_map.py`](../construction/generate_map.py), not the
dataset-specific `main()` retained in the research modules.

```text
RGB semantic PCD + TUM poses
    -> save_stopline_points.main: stop-line-based road segmentation
    -> process_lane_segments.process_lane_segment: lane fitting
    -> save_segment_shapefile_with_numbers: original transition export logic
    -> concatenation of segment features -> lane_map.shp
```

The entry also accepts already segmented cloud/pose pairs. Incomplete-tail
files are included, not silently discarded. No cross-road fusion or junction
connectivity is inferred by concatenating features.

| Module / function | Responsibility |
| --- | --- |
| `save_stopline_points.py` | Trajectory windows, semantic stop-line checks and delayed truncation |
| `process_lane_segment` | Pose/PCA direction, empirical fitting-branch selection and reconstruction |
| `classify_dashed_lane_with_ends` | Extend dashed support with eligible end-region observations |
| `fit_lane_segments_adaptive`, `fit_multi_lanes` | Block fitting and lateral lane separation |
| `sliding_window_lane_fitting` | Local-window lane fitting |
| `classify_lane_type_by_semantic_ratio` | Semantic-ratio dashed/solid classification |
| `add_shorter_left_solid_lane`, `add_shorter_right_solid_lane` | Borrow adjacent-line coordinates for merge/split geometry |
| `save_segment_shapefile_with_numbers` | Preserve the core's dashed split/shared-coordinate construction |
| `road_structure_analyzer.py` | Original supplementary turn analysis; retained, not automatically substituted into the pipeline |

Cloud colors are normalized RGB: green `(0,1,0)` solid, red `(1,0,0)` dashed,
yellow `(1,1,0)` arrow and blue `(0,0,1)` stop marking. Pose rows must be
`timestamp x y z qx qy qz qw`, with finite values and nonzero quaternions.
Cloud and trajectory must already share the same metric frame. The entry does
not estimate registration, align unrelated datasets, or infer a missing CRS.

For metric projected inputs, `--input-crs` preserves coordinates. For ENU inputs,
`--enu-origin LAT LON HEIGHT` applies ENU -> ECEF -> WGS84, with longitude first
in Shapefile coordinates. This explicit conversion replaces the old entry's
hardcoded origin and incorrect assignment of EPSG:4326 to local x/y.
It is not claimed to be bit-for-bit identical to historical exports; the old
project contains a spherical `geographiclib` shim as well as geodesic-based code.

Export keeps `id`, `segment_id`, `left_lane`, `right_lane` where supplied by the
original function, adds a unique `edge_id`, and normalizes `type=lane` and
`lane_type=solid|dashed` for geometry evaluation. Original lane numbering is
retained as legacy metadata, not claimed to be a validated global lane-count
attribute. `source`, `target` and `LaneWidth` are not inferred by this entry.

Outputs are written only to a new directory. Each run includes `generation.log`
and `run_manifest.json` with input/code hashes, seed, parameters, segment status
and omitted stages. A segment failure prevents publication of a normal map.
`--allow-partial` explicitly saves `lane_map_partial.shp` (or
`vector_map_partial.shp` with arrows enabled) and exits with code 2.
Fitting may reject sparse or strongly curved inputs under the configured
thresholds; a rejected segment is reported as a failure, not silently omitted.

## Arrow Reconstruction

`--with-arrows` enables `construction/arrow_generation.py` inside the lane
generation entry. The optional core callback runs at the same stages as the
recovered arrow implementation: immediately after window fitting (`fitting_type=0`)
or after lane-length normalization in the other fitting branch (`fitting_type=1`).
The lane fitter's return tuple and reconstruction thresholds are unchanged.

The retained processing rules are:

- For fitting type 1, remove arrow points within 0.5 m of reconstructed lane
  polylines. Type 0 does not apply this removal.
- Cluster XYZ points with Open3D DBSCAN, `eps=0.9 m`, `min_points=800`.
- Retain clusters using the original lane proximity and two-sided 5 m checks.
  These are heuristics, not a polygon-containment proof: the first test uses
  distance to line extensions; the second checks distances around offset points.
- Group cluster PCA directions with a 15-degree parallel/antiparallel threshold.
  Rectangle orientation still follows the segment's trajectory direction.
- Construct projected-extents rectangles with a 1 m minimum length and width.
  If width exceeds 2 m or length exceeds 10 m, use the original secondary
  DBSCAN (`eps=0.6 m`, `min_points=100`) and largest-cluster size refinement.

The original box center is the midpoint of directional extrema, whereas the
saved `center` is the cluster mean. Secondary clustering refines size but does
not recenter the box in that source. These conventions are retained rather than
silently changing historical geometry. Export stores cluster means separately
as `cluster_x/y/z`; XY uses the map CRS and Z follows the coordinate conversion.
The geometry evaluator continues to measure polygon-centroid error.

Arrow export converts each box to a closed XY LineString so it can share one
Shapefile with lanes. `vector_map.shp` contains `type=lane|arrow`, unique `edge_id`,
segment metadata, and per-segment `arrow_id`/`group_id` for arrows. No cross-road
arrow deduplication or junction inference is added. GUI rendering is disabled;
invalid data and failed arrow stages cannot silently become successful maps.

Input adaptation normalizes nested lane-window arrays. Single-window two-point
lists and deeper nested arrays can now be used where the original list parser
skipped them or failed. This is an explicit compatibility change; retaining
numeric thresholds does not imply identical detections for every input layout.

## Stage Boundaries

| Stage | Public source | Inputs and scope |
| --- | --- | --- |
| Image/LiDAR semantic mapping | `construction/semantic/` | Includes inference configuration; requires images, LiDAR, poses, matching calibration, car mask and trained weights |
| Lane fitting and export | `construction/generate_map.py` and core modules | Registered semantic clouds and trajectories; exports lane geometry with subtype metadata |
| Arrow reconstruction and export | `construction/arrow_generation.py`, `generate_map.py --with-arrows` | Registered arrow points and reconstructed lane context; exports rotated closed rectangle linework |
| Arrow inspection | `construction/semantic/arrow_vectorized.py` | Clustering and axis-aligned box inspection; not the final oriented-arrow Shapefile exporter |
| Map merging | `postprocessing/fusion_multishp.py` | Existing lane/arrow Shapefiles; preserves available attributes, does not infer junction connectivity |
| Lane attribute preparation | `postprocessing/prepare_lane_attributes.py` | Adapts lane-only construction output to the width schema; preserves original IDs and leaves missing node IDs null |
| Attribute computation | `postprocessing/cal_shp_lane_info*.py` | Width variant accepts the adapter output; the separate label variant has different gate semantics |
| Coordinate conversion | `postprocessing/` conversion utilities | Explicit source/target coordinate conventions; conversion is not registration |

The arrow stage is connected to lane reconstruction by `--with-arrows`; merging
pre-existing boxes remains a separate postprocessing operation. The attribute
adapter accepts lane-only output and supplies the width-input schema, not missing
topology. Existing `source`/`target` values are not evidence
that a merge operation recovered junction topology. Full sensor-to-attributed-map
execution and exact historical final-map reproduction have not been verified.

See [semantic source instructions](../construction/semantic/README.md) for
model/data preparation and [postprocessing instructions](../postprocessing/README.md)
for commands, field order, numeric gate assumptions and CRS restrictions.
The 3D-to-2D converter retains endpoint heights; it is not a substitute
for geodetic coordinate transformation.

## Publication Selection

The release includes the construction algorithms, associated semantic helpers,
selected map postprocessing tools and three evaluation entry points. Dataset-
specific adapters with incompatible core signatures are excluded, as are GUI
launchers, PLC/Modbus utilities, temporary experiments, duplicate plotting
variants, model weights, local calibration/mask files and measured results.
Exclusion from this release does not delete these files from the research
projects. Upstream licensing information is in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Runtime Verification

Lane generation was tested on existing data, using the
Python 3.8 `lane3d` environment, Open3D 0.19.0 and scikit-learn 1.3.2. Local
results are under `results/construction_check/` and excluded from the public
source archive. They are execution checks, not additional paper measurements.

| Case | Observed result |
| --- | --- |
| Bundled semantic configuration, existing checkpoint, one image on CPU | Valid 720x1280 label array |
| One synchronized image/LiDAR/pose frame through semantic fusion | 5,149 finite colored points exported and read back from PCD |
| KAIST segment 009, projected-input entry | 3 valid dashed line features, EPSG:32652 |
| Same cloud/poses, stop-line segmentation entry | 3 valid dashed line features, EPSG:32652 |
| Original-dataset segment 001, explicit ENU origin | 65 valid line features: 62 solid, 3 dashed, EPSG:4326 |
| Same segment with arrow stage enabled | Identical 65 lane geometries and attributes; zero arrow detections because all 5,385 arrow points were DBSCAN noise under the original density threshold |
| Existing dense `segments_arrow/lane_segment_001.pcd`, arrow stage enabled | 33 lane features and 6 closed arrow rectangles, EPSG:4326; valid geometry, finite cluster centers and unique edge IDs verified after Shapefile read-back |
| Same 65-feature lane export, schema adaptation and width calculation | 65 valid features in EPSG:32650; IDs retained; 24 nonzero widths; missing node fields remain null |
| Original-dataset segment 002 | Failed; no map published. Re-clustering returned one line but the original window fitter indexes `results_lines[1]` |

The last failure remains unresolved because a fallback changes which lines
are inferred. It is not concealed by treating the failed road as absent.
Its traceback and parameters are retained in the run log/manifest. Therefore
the current source must not be described as successfully generating all roads.
The uninitialized diagnostic-variable error encountered earlier was repaired
separately, without changing fitted geometry or fitting thresholds.

## Evaluation Source

- `evaluation_trajmap/geometry/local/evaluate_vector_geometry.py`: metric
  projection, arc-length sampling, directional nearest-neighbor distances and
  polygon IoU/center matching. Overall lane matching ignores dashed/solid labels.
- `evaluation_trajmap/topology/evaluate_topology_metrics.py`: local directed
  connectivity and implementation-specific path/fragmentation diagnostics.
- `evaluation_trajmap/geometry/global/topology_eval.py`: global graph compression,
  node matching and path comparison. Its historical directory name is retained.

Metric conventions and limitations are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
Generation changes do not recompute or replace the historical experiment results.
