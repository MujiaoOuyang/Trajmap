# Metric and Reproducibility Notes

## Lane Distances and Coverage

For each sampled point x in GT, let d_GP(x) be its distance to the nearest point
in prediction. Define d_PG independently in the reverse direction. The sets
need not have equal sizes, and the nearest-neighbor relation need not be mutual.

For each direction, `mean_m` is the arithmetic mean, `median_m` is NumPy's median,
and `p80_m`, `p90_m`, `p95_m` use `numpy.quantile` at 0.80, 0.90, 0.95 with its
default linear interpolation. `coverage_within_rm` is `mean(distance <= r)`.
`point_cloud_overlap.*.ratio_within_rm` computes the same point coverage when
evaluated on the same features, spacing and direction.

```text
mean_GP = sum(d_GP) / N_GT
mean_PG = sum(d_PG) / N_pred
symmetric_cd_m = mean_GP + mean_PG
table_mean_m = (mean_GP + mean_PG) / 2
Cov@r = (mean(d_GP <= r) + mean(d_PG <= r)) / 2
Cov-hard = (Cov@0.2 + Cov@0.5 + Cov@1.0) / 3
Cov-easy = (Cov@0.5 + Cov@1.0 + Cov@1.5) / 3
```

Coverage in JSON is a fraction; multiply by 100 for percent. The paper-summary
convention averages the two directional median/percentile statistics. This is
not the same as taking a quantile after concatenating both distance arrays.
Always state which convention is used.

For a line of length L and requested spacing h, samples occur at k*L/ceil(L/h),
including both endpoints. Actual spacing is at most h and depends on L. Repeated
endpoints of separate features remain repeated; results are not fully invariant
to feature fragmentation. `num_gt` and `num_pred` count line parts, whereas
`num_*_sample_points` count samples. Equal total lengths do not imply equal
feature counts or equal sample counts.

`lane_overall` permits cross-subtype nearest neighbors and can include unknown
subtypes. Class-specific evaluation only permits same-subtype candidates. All
can consequently be better than both Solid and Dashed without an averaging bug.

Legacy `ap_precision` is TP/(TP+FP) at a threshold. It does not integrate a
confidence-ranked precision-recall curve. Hard/easy AP-labelled fields average
feature-match precision, not geometric coverage; do not label them standard AP,
mAP, or Cov-hard/easy in a paper.

## Arrows

Polygon intersections/unions account for rectangle rotation. Center errors use
polygon centroids in the common metric CRS, not an independently loaded point
cloud cluster-center attribute. Ensure the intended centers agree upstream.
The optional construction arrow stage saves the cluster mean in `cluster_x/y/z`
but preserves the original extrema-based box center; the two can differ. These
extra fields do not change the evaluator's polygon-centroid definition.

IoU and center thresholds have separate greedy one-to-one match sets, not a
joint optimal assignment. Candidate pairs have a finite center-distance search
gate. Matched-pair IoU and center summaries do not penalize every unmatched box;
report TP/FP/FN and threshold-specific recall alongside them. Very small polygons
are excluded by `--arrow-min-area` (default 0.5 square meters).

## Local Topology

Do not interpret local graph nodes as junction nodes: each node represents a
locally segmented lane-line element. Candidate matching uses geometry, direction
and endpoints. The precision/recall workflows include different matching
constraints and path recovery; they are not a single universal graph-edit score.

- **Path score:** `shortest_path` and `shortest_length` use FIFO traversal with
  early return, so the first returned feasible path need not minimize weighted
  length. Historical APLS-labelled outputs are implementation-specific. Replacing
  this search with Dijkstra requires a metric version change and recomputation.
- **BR:** when both sets are nonempty,
  `1 - min(1, mean_pred_segment_length / mean_gt_segment_length)`. This is a
  fragmentation proxy. It can be zero even when connections are missing. The
  separately reported `segmental_coverage` is not BR's defining term.
- **DR:** sum of lengths marked redundant by prediction-only proximity/heading
  checks, divided by total predicted segment length. It is not GT-based FP rate.

The manuscript protocol uses 5/10/20 m local segmentation (10 m primary), 1 m
sampling, 2 m distance and endpoint thresholds, 1 m cross-chain endpoint
threshold, 0.1 m chain merging, 45-degree direction threshold, and 0.3 relative
length-error tolerance. Shared trajectory-normal segmentation requires the
trajectory with its correct coordinate convention. Simply using all CLI defaults
does not reproduce this configuration or the original spatial exclusions.

## Global Topology

`calc_global_apls` uses weighted directed Dijkstra paths between matched nodes.
The reported global score averages the two directional raw path similarities
and multiplies by the minimum matched-node fraction of the two graphs.

The matching parameter `--match-tolerance` is not a hard positional rejection
radius: candidate searches also use a larger radius/nearest-candidate fallback
and topology-based refinement. Compression uses a directed simple graph, can
collapse parallel paths, and does not guarantee preserving every isolated cycle
or explicitly protected node. Do not claim unconditional topology preservation
from these implementation rules alone.

The historical global result version reporting 97.94% Topo-F1 used an explicit
prediction node correction in a separate local runner. That correction is not
enabled by this public evaluator. The value must not be presented as a verified
fresh, unmodified run of this release.

## Protocol and Data Availability

No measured paper results or real-world maps/trajectories are included in the
public allowlist. For exact experiment reproduction, the authors still need to
freeze and release, or document access restrictions for:

1. GT and prediction Shapefile bundles, their hashes and class mappings;
2. spatial masks and exclusions (including intersections/transition regions);
3. trajectory frame/origin and any alignment procedure;
4. exact command lines, dependency versions, method versions and random seeds;
5. preprocessing, attribute preparation and any manual corrections.

Neither geometry evaluation nor the archive builder independently establishes
GT annotation independence or baseline comparability. Use the same extent,
sampling, metric convention and allowed labels across methods. Do not fill
prediction attributes from GT before reporting attribute accuracy.

## Environment and Verification

The public geometry tests were run in two existing environments:

| Dependency | Geometry-only check | Geometry and global CLI/plot checks |
| --- | --- | --- |
| Python | 3.11.11 | 3.8.20 |
| NumPy | 2.2.6 | 1.24.4 |
| SciPy | 1.15.3 | 1.10.1 |
| GeoPandas | 1.0.1 | 0.13.2 |
| Shapely | 2.1.1 | 1.8.4 |
| pyproj | 3.7.1 | 3.5.0 |
| NetworkX | 3.4.2 | 3.1 |
| Matplotlib | Not installed | 3.7.5 |

These are observed verification environments, not a claim of support for every
combination or a recommendation to adopt an older Python version. The dependency
list in `requirements.txt` is not a fully tested lockfile.

`tests/public` contains source-only packaging checks and synthetic geometry
checks. Geometry tests are explicitly skipped when required GIS imports are
unavailable; a skipped test is not validation of that evaluator. The global CLI
test file documents its dependency handling. Construction tests additionally
require the construction dependencies. Three existing-data lane-generation
checks succeeded, while another road still fails inside the original window
fitter; see [SOURCE_GUIDE.md](SOURCE_GUIDE.md) for exact inputs and scope.
Full sensor-to-attributed-map generation remains unverified; no claim of
frozen-environment or full-paper reproduction is made.

The local workspace may contain additional manuscript/data-dependent tests.
They are not in the release manifest and are not required to run the public tests.

### Source-Package Checks (2026-11-26)

- Public suite: 86 tests, 85 passed and 1 GDAL-dependent test skipped, using
  Python 3.8 `lane3d` with isolated PyShp 2.3.1/Rtree 1.3.0 dependencies.
- Global topology suite: 18 tests passed in `lane3d`.
- Independent system Python 3.8.10/GDAL 3.4.3 CLI check passed for 3D-to-2D
  conversion, multipart geometry, XY preservation, endpoint heights, CRS,
  field order and overwrite rejection. This is separate from the skipped test.
- Semantic tests cover synthetic helpers, import/help isolation, calibration,
  mask recovery, PCD reading and self-contained inference configuration.
- The bundled configuration matched the checkpoint's embedded network and
  preprocessing settings, apart from disabled backbone pretraining. CPU model
  loading and one real-image inference succeeded. One synchronized sensor-frame
  fusion run exported 5,149 finite colored points; a full raw-sensor route was
  not reconstructed during this check.
- The schema adapter and width calculator processed an existing 65-feature lane
  export, preserving original IDs and segment metadata. Output widths were
  finite, with 24 nonzero values; this is execution validation, not a width-
  accuracy measurement. Unknown graph-node fields remained null.
- Arrow generation tests cover dense synthetic clusters, rotated boxes, empty
  detections, retained filtering rules, nested lane inputs, headless execution,
  separate cluster/box centers and coordinate conversion. The CLI rejects failed
  arrow stages rather than publishing them as complete vector maps.
- On one existing dense semantic-cloud/trajectory segment, the lane-plus-arrow
  entry exported 33 lane features and 6 closed arrow rectangles in EPSG:4326.
  Shapefile read-back confirmed valid geometries, finite cluster-center fields
  and unique edge IDs. Another, sparser segment produced zero arrows under the
  original DBSCAN threshold and retained its previous 65 lane geometries and
  attributes exactly. These checks do not measure arrow detection accuracy.

These are software checks, not additional experimental results or ablations.
The requirements lists are not evidence of a clean installation on a new system.
