# TrajMap

Source and evaluation code accompanying **TrajMap: Trajectory-Guided
Topology-Preserving Global Lightweight HD Map Generation**.

TrajMap provides tools for **trajectory-guided map generation**, **map
postprocessing**, and evaluation of **geometric accuracy** and **topological
consistency**.

The stages have separate entry points and explicit input requirements. The
semantic-PCD/TUM entry generates lane-line maps and optional arrow rectangles; postprocessing handles existing
map elements and attributes. This release does not claim a verified one-command
reproduction of every historical paper result. See the
[source guide](docs/SOURCE_GUIDE.md) for each stage's inputs and verification scope.

## Evaluation Metrics

### 1. Geometry Evaluation

Lane lines and arrow markings are evaluated separately.

#### Lane Lines

Each input line is sampled at uniform arc-length intervals **no larger than**
the requested spacing. Both endpoints are included. GT and prediction may have
different sample counts; bidirectional nearest-neighbor searches do not require
point-to-point correspondence or equal-sized sets.

The outputs include:

- Mean and median nearest-neighbor distance, in meters;
- P80, P90 and P95 distance percentiles;
- `Cov@r`: fraction of samples whose nearest-neighbor distance is at most `r`;
- `Cov-hard`: average coverage at 0.2 m, 0.5 m and 1.0 m;
- `Cov-easy`: average coverage at 0.5 m, 1.0 m and 1.5 m.

The JSON reports GT-to-prediction and prediction-to-GT separately. For a single
bidirectional coverage value, average the two directions first. `Cov-hard` and
`Cov-easy` are derived from these coverages, **not** from the legacy `*_hard_ap`
and `*_easy_ap` fields.

**Distance convention:** `symmetric_cd_m` is the **sum** of the two directional
means. The bidirectional **Mean** used in the manuscript table is their average,
so `Mean = symmetric_cd_m / 2`. These are different conventions, not different
distance units. See [metric definitions](docs/REPRODUCIBILITY.md).

`lane_overall` ignores dashed/solid distinctions when finding nearest neighbors;
`lane_dashed` and `lane_solid` restrict candidates to the same subtype. Overall
error is therefore not a weighted average of the class-specific errors.

#### Arrow Markings

Arrow rectangle linework is converted to polygons and evaluated using:

- Matched counts and threshold-specific precision, recall and F1;
- Rotated polygon intersection over union (IoU);
- Center translation error in meters.

IoU matching and center-distance matching use separate greedy one-to-one
assignments. The evaluated center is the polygon centroid. It represents the
arrow clustering center only when the input box was constructed with that
convention. Report a matching ratio with its threshold and denominator, and
report matched-pair errors together with unmatched counts.

### 2. Local Topology Evaluation

Local evaluation constructs a directed graph of lane-line segments and measures
nearby connectivity. Outputs include topology precision, recall and F1,
implementation-specific directed APLS at 30 m, 50 m and 80 m, Break Rate (BR),
and Duplicate Rate (DR).

Higher precision, recall, F1 and path scores are better. Lower BR and DR are
better under this implementation's definitions. BR is a segment-length
fragmentation proxy, not a direct count of disconnected paths; DR measures
redundancy within the prediction. The existing path search is not guaranteed
to return weighted shortest paths. These implementation details are documented
in [reproducibility notes](docs/REPRODUCIBILITY.md).

### 3. Global Topology Evaluation

Global evaluation constructs and compresses directed lane-line graphs, retaining
topology-critical junction, transition and boundary structure where supported
by the input fields and compression rules. Outputs include topology precision,
recall and F1, a directed APLS-like score, and node matching coverage.

The final APLS-like score includes path consistency and topology-node coverage.
This is an implementation-specific lane-line graph metric, not a guarantee of
legal vehicle-turn connectivity. Compression and matching limitations are listed
in the reproducibility notes.

## Repository Structure

```text
trajmap/
|-- README.md
|-- requirements.txt
|-- requirements-construction.txt
|-- release_manifest.json
|-- source_provenance.json
|-- THIRD_PARTY_NOTICES.md
|-- licenses/
|-- construction/
|   |-- semantic/
|   |-- generate_map.py
|   |-- arrow_generation.py
|   |-- process_lane_segments.py
|   |-- save_stopline_points.py
|   `-- road_structure_analyzer.py
|-- postprocessing/
|-- evaluation_trajmap/
|   |-- geometry/
|   |   |-- local/evaluate_vector_geometry.py
|   |   `-- global/topology_eval.py
|   `-- topology/evaluate_topology_metrics.py
|-- docs/
|   |-- SOURCE_GUIDE.md
|   `-- REPRODUCIBILITY.md
|-- scripts/build_release.py
`-- tests/
    |-- public/
    `-- test_global_topology_cli.py
```

The historical `geometry/global/` path contains the **global topology evaluator**,
not a second geometry metric. It is retained for compatibility. The local research
workspace also contains scripts and results not included in this release;
`release_manifest.json` is the authoritative publication file list.

## Source Code Overview

| Paper component | Source | Main responsibilities |
| --- | --- | --- |
| Semantic observation preparation | `construction/semantic/` | Camera/LiDAR semantic projection, arrow-mask processing and supporting prediction utilities |
| Lane generation entry | `construction/generate_map.py` | Explicit inputs/CRS, headless execution, original per-segment export, manifest and failure handling |
| Road segmentation | `construction/save_stopline_points.py` | Trajectory windows and stop-line-triggered segmentation |
| Trajectory-guided vectorization | `construction/process_lane_segments.py` | Semantic point selection, trajectory direction, RANSAC fitting, local windows, lane ordering and per-segment export |
| Arrow reconstruction | `construction/arrow_generation.py` | Semantic-point filtering, DBSCAN clusters, lane-context checks and trajectory-oriented rectangles |
| Map postprocessing | `postprocessing/` | Existing Shapefile merging, arrow collection, attribute processing and coordinate conversion |
| Lane and arrow geometry | `evaluation_trajmap/geometry/local/evaluate_vector_geometry.py` | Metric projection, line sampling, nearest-neighbor distances, polygon IoU and center matching |
| Local directed connectivity | `evaluation_trajmap/topology/evaluate_topology_metrics.py` | Chain assembly, segmentation, directed adjacency, edge/path matching and fragmentation diagnostics |
| Global graph comparison | `evaluation_trajmap/geometry/global/topology_eval.py` | Graph construction from fields or endpoints, compression, node matching, path comparison and diagnostic plots |

See [SOURCE_GUIDE.md](docs/SOURCE_GUIDE.md) for function-level explanations,
input/output conventions, and the boundary between reconstruction and evaluation.
The available source does not implement all manuscript stages in one command.

## Installation

Use a dedicated Python environment. Checks have used Python 3.8 and 3.11 in
different environments; dependency details and verification limits are
recorded in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

```bash
git clone https://github.com/MujiaoOuyang/trajmap.git
cd trajmap
python -m pip install -r requirements.txt
```

For semantic-PCD/TUM lane generation, install the construction dependencies:

```bash
python -m pip install -r requirements-construction.txt
```

For map fusion and attribute tools, install
`postprocessing/requirements-postprocessing.txt`. See the
[postprocessing guide](postprocessing/README.md) for the separate GDAL requirement
of the 3D-to-2D converter. Semantic preparation dependencies are listed in
`requirements-semantic.txt`; the inference framework and compatible weights
must be supplied separately. A self-contained inference configuration is included,
as described in the
[semantic source guide](construction/semantic/README.md).

The dependency files are installation lists, not a frozen publication environment.
Full sensor-to-attributed-map reproduction remains unverified; see the source
guide for stage-specific dependencies, data requirements and execution limits.

## Input Data

Supply complete Shapefile bundles, including `.shp`, `.shx`, `.dbf` and `.prj`
files. Preserve `.cpg` when available. A `.cpg` file alone is not a map.

The CRS must be defined. Geometry/local evaluation chooses a common projected
CRS; **projected coordinates must use meters**. Global evaluation requires an
explicit projected meter CRS appropriate to the study area.

A recommended geometry schema is:

| Field | Values / meaning |
| --- | --- |
| `type` | `lane` or `arrow` |
| `lane_type` | `dashed`, `solid`, `left solid`, `right solid`; optional for overall geometry |
| `source`, `target` | Node identifiers for explicit directed global graph connectivity |

Use `--gt-class-field`, `--pred-class-field`, `--gt-class-map` and
`--pred-class-map` to adapt different class schemas. Do not assume numeric labels
have the same meaning across methods. For example, when verified against your
data, `--gt-class-map "0=lane,1=arrow"` maps numeric GT labels.

For the geometry entry point, avoid the legacy `--ignore-class` and `--classes`
options; their behavior does not match the current classified workflow. Use
explicit class mappings instead. Arrow boxes should be represented as one
polygon/closed line or grouped multiline feature per arrow.

## Usage

### Semantic Point Cloud Preparation

The image/LiDAR projection entry requires compatible model weights,
calibration, car mask and synchronized dataset. Its argument help does not load
model weights or initialize CUDA:

```bash
python construction/semantic/myConcateTemp.py --help
```

Use `--model-config construction/semantic/configs/segformer_whu_inference.py`
with the corresponding WHU SegFormer checkpoint. The configuration does not
depend on files installed outside this repository.

The [semantic guide](construction/semantic/README.md) describes the input layout,
full command line and calibration conventions.
`construction/semantic/arrow_vectorized.py` is a separate GUI inspection tool,
not an arrow-Shapefile export command.

### Lane Map Generation

Already segmented semantic clouds and paired TUM trajectories:

```bash
python construction/generate_map.py \
  --segment-dir path/to/segments \
  --enu-origin 32.022252 118.726685 8.135 \
  --output-dir results/generated_lanes
```

The origin above is for the original Nanjing ENU frame only. For cloud and pose
coordinates already in a projected meter CRS, replace `--enu-origin ...` with
`--input-crs EPSG:32650` (using the actual CRS, not a guessed one).

To include the existing stop-line segmentation stage:

```bash
python construction/generate_map.py \
  --pcd path/to/semantic_global.pcd --poses path/to/trajectory_tum.txt \
  --enu-origin 32.022252 118.726685 8.135 \
  --output-dir results/generated_from_cloud
```

The output directory must **not exist**. Output includes `lane_map.shp`, its
sidecars, `generation.log` and `run_manifest.json`. This entry does not export
junction virtual nodes or lane-width attributes. Failed
segments prevent a normal map export; details remain in the manifest.

### Lane and Arrow Map Generation

Enable arrow reconstruction from yellow semantic points in the same cloud:

```bash
python construction/generate_map.py \
  --segment-dir path/to/segments \
  --enu-origin 32.022252 118.726685 8.135 \
  --with-arrows --output-dir results/generated_vectors
```

This writes `vector_map.shp` instead of `lane_map.shp`. Each arrow is one closed
rectangle LineString with `type=arrow`; lane lines retain `type=lane` and their
`lane_type`. A single line layer avoids mixing polygon and line Shapefile types.
The manifest records lane and arrow counts separately, including zero detections.

`cluster_x`, `cluster_y`, `cluster_z` retain the point-cluster mean after coordinate
conversion. It is not necessarily the rectangle centroid. The current geometry
evaluator uses the rectangle centroid, not these attributes. Original density
thresholds and lane-context filters are retained; sparse arrow observations may
produce no boxes. See the [arrow source details](docs/SOURCE_GUIDE.md#arrow-reconstruction).
Without `--with-arrows`, the lane-only workflow is unchanged and remains suitable
for the lane attribute adapter below.

### Shapefile Postprocessing

Merge two existing maps, then optionally collect arrow features from a directory
of section maps. Inputs must have matching CRS; merging does not align maps or
create missing connectivity:

```bash
mkdir -p results/postprocessing
python postprocessing/fusion_multishp.py merge \
  --input1 path/to/map_a.shp --input2 path/to/map_b.shp \
  --output results/postprocessing/merged.shp
python postprocessing/fusion_multishp.py arrows \
  --base results/postprocessing/merged.shp --arrow-dir path/to/section_maps \
  --output results/postprocessing/with_arrows.shp
```

Attribute-processing scripts retain the original field-order requirements; they
are not generic schema converters. Consult the source guide before using them.
Postprocessing refuses existing output basenames, including Shapefile sidecars.

### Lane Attribute Export

For a lane-only map generated by `construction/generate_map.py`, prepare the
legacy width-input schema, then run the width calculation:

```bash
mkdir -p results/attributes
python postprocessing/prepare_lane_attributes.py \
  --input results/generated_lanes/lane_map.shp \
  --angle-threshold 10 --output results/attributes/width_input.shp
python postprocessing/cal_shp_lane_info.py \
  --input results/attributes/width_input.shp \
  --output results/attributes/attributed_lanes.shp
```

Choose the original method's 10- or 60-degree angle gate explicitly. The adapter
preserves the original ID as `orig_id`, retains `edge_id` and `segment_id`, and
assigns unique scan IDs. It does not infer graph nodes: absent `source`/`target`
remain null. This workflow currently accepts geographic coordinates or
EPSG:32650, outputs EPSG:32650, and is intended for the study region. It does not
support arrow features or the separate lane-number/label variant. Zero widths
can denote an unresolved neighbor under the original algorithm, not physical
zero-width lanes. See [postprocessing details](postprocessing/README.md).

Create output directories first; geometry/local evaluation writes to the supplied
paths and can overwrite existing output files. Keep GT and prediction inputs
separate from all outputs.

```bash
mkdir -p results/geometry results/local
```

### Geometry Evaluation

```bash
python evaluation_trajmap/geometry/local/evaluate_vector_geometry.py \
  --gt path/to/ground_truth.shp --pred path/to/prediction.shp \
  --class-field type --lane-type-field lane_type --sample-spacing 0.2 \
  --output-json results/geometry/metrics.json \
  --output-csv results/geometry/metrics.csv
```

### Local Topology Evaluation

This example uses arc-length segmentation and requires no trajectory:

```bash
python evaluation_trajmap/topology/evaluate_topology_metrics.py \
  --gt path/to/ground_truth.shp --pred path/to/prediction.shp \
  --segmentation-mode line --segment-lengths 5 10 20 \
  --primary-segment-length 10 --angle-threshold 45 \
  --output-json results/local/metrics.json \
  --output-csv results/local/metrics.csv \
  --output-plot results/local/matching.png
```

For the manuscript's trajectory-normal protocol, change `--segmentation-mode`
to `trajectory-normal`, supply `--trajectory-tum`, and explicitly set
`--lat-ref`, `--lon-ref`, `--h-ref` to the actual ENU origin. If TUM x/y are already
in the map's projected CRS, use `--trajectory-already-map-crs`. These two
protocols are not interchangeable. The legacy default lengths are 20/30/50 m,
so the explicit 5/10/20 m arguments above matter.

### Global Topology Evaluation

```bash
python evaluation_trajmap/geometry/global/topology_eval.py \
  --gt path/to/ground_truth_topology.shp \
  --pred path/to/prediction_topology.shp \
  --target-crs EPSG:32650 --output-dir results/global
```

`EPSG:32650` is an example for the study region, not a universal projection.
Use a new or empty output directory. This entry point does not apply the
dataset-specific manual node correction used by one historical result version.

### Tests

Tests create synthetic geometry in temporary directories; private survey data
are not needed. The full public test suite also requires
`requirements-construction.txt` (including Open3D). Real-data lane-generation
checks and the remaining failing road are documented in the source guide.

```bash
python -B -m unittest discover -s tests/public -v
python -B -m unittest discover -s tests -p test_global_topology_cli.py -v
```

## Output

Geometry/local evaluation exports JSON and CSV; topology evaluation also produces
diagnostic figures and reports. Keep generated outputs outside source directories.
Missing classes or empty matches must not be interpreted as measured zero error.

| Category | Metrics |
| --- | --- |
| Lane geometry | Directional Mean, Median, P80/P90/P95, Cov@r, summed CD |
| Arrow geometry | Matched counts, threshold precision/recall/F1, IoU, center error |
| Local topology | Precision, Recall, F1, implementation-specific APLS, BR, DR |
| Global topology | Precision, Recall, F1, directed APLS-like score, node coverage |

Legacy fields named `ap_precision`, `*_ap` or `mean_ap_*` report fixed-threshold
precision or averages of such precision. They are **not confidence-ranked AP or
mAP**, and are not the point coverage metrics above.

## Preparing a Public Archive

```bash
python scripts/build_release.py --directory dist/trajmap
```

This creates the clean upload directory `dist/trajmap/` and matching
`dist/trajmap.zip` from the explicit allowlist, including SHA-256
checksums. It refuses existing destinations. Raw maps,
trajectories, point clouds, manuscript attachments and historical experimental
outputs are excluded. Upload the **contents of the clean directory**, not the
research workspace. The builder does not upload anything. Review file contents,
source ownership, data rights and dependency licenses before sharing.

## License and Citation

The upstream HDMap MIT notice is preserved in `licenses/HDMap-MIT.txt`.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A repository-wide license
for the authors' additions has not yet been selected; preserving an upstream
notice does not assign that license to all source, models or data.

If this toolkit is useful in your research, please cite the TrajMap paper once
its bibliographic information is available. The working title is given above;
no publication venue, year or DOI is asserted here.

## Contact

For questions or suggestions, please open an issue in the repository.
