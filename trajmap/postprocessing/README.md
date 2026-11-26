# Shapefile Postprocessing

These tools merge existing map features, calculate lane attributes, and
flatten existing 3D geometry. They do not detect arrows,
create junctions, repair topology, fit lanes, or generate source/target IDs.
The commands below are independent tools, not an automatically compatible
end-to-end pipeline.

## Files

| File | Purpose |
| --- | --- |
| `fusion_multishp.py` | Strict two-file merge; append existing arrow features |
| `cal_shp_lane_info.py` | Lane count, label, width, and arrow-center conversion |
| `prepare_lane_attributes.py` | Adapt lane-only construction output to the width-input schema |
| `cal_shp_lane_info_lanenumber_label.py` | Lane count/label variant using Rtree |
| `convert_3d_to_2d.py` | Flatten geometry; replace center with endpoint heights |
| `_shapefile_io.py` | Output, CRS, and positional-schema checks |
| `__init__.py` | Importable package |

## Dependencies

From the repository root, in a suitable Python environment:

```sh
python -m pip install -r postprocessing/requirements-postprocessing.txt
```

The requirements file records versions actually loaded during the synthetic
tests: Python 3.8 `lane3d` plus isolated temporary PyShp/Rtree dependencies.
PyShp/Rtree were not installed in `lane3d`; no other checked Conda or system
installation supplied PyShp. This is not a clean-install validation or a claim
of compatibility with every newer Python release. GDAL was tested separately
with system Python and is not part of this pinned environment.

- Strict fusion: PyShp and PyProj.
- Arrow fusion: GeoPandas, Pandas, PyProj, and its Fiona/Shapely dependencies.
- Width attributes: PyShp and PyProj.
- Label variant: additionally Rtree and its native spatial-index library.
- 3D conversion: additionally `osgeo.ogr` from GDAL Python bindings matched to
  the environment's native GDAL library. GDAL is not pinned in the pip file.
  Fiona's bundled GDAL alone does not provide `osgeo`.

All optional libraries are imported only when the corresponding operation is
called. Importing these modules or running `--help` does not open input data,
write outputs, modify PROJ configuration, or require optional GIS libraries.

## Commands

Run from the repository root. Output directories must already exist. Every
output basename must be new, including its `.shp`, `.shx`, `.dbf`, `.prj`, and
other existing sidecars. There is no overwrite mode.

```sh
python -m postprocessing.fusion_multishp merge --input1 input/a.shp --input2 input/b.shp --output results/merged.shp
python -m postprocessing.fusion_multishp arrows --base input/base.shp --arrow-dir input/sections --output results/with_arrows.shp
python -m postprocessing.cal_shp_lane_info --input input/width_ready.shp --output results/attributes.shp
python -m postprocessing.cal_shp_lane_info_lanenumber_label --input input/label_ready.shp --output results/labels.shp
python -m postprocessing.convert_3d_to_2d --input input/map3d.shp --output results/map2d.shp
```

Direct scripts also work, for example
`python postprocessing/fusion_multishp.py merge --help`.
The callable entrypoints are `twofileFusion_strict(input1, input2, output)`,
`extract_and_merge_arrows(base, folder, output)`, `process(input, output,
input_crs=None)` in each attribute module (`input_crs` is keyword-only), and
`convert_3d_to_2d(input, output)`.

## Preserved Behavior and Preconditions

### Fusion

Both merge inputs must be 2D polylines with matching declared `.prj` CRSs.
Coordinates are not transformed. The original forced EPSG:4326 label is
replaced by the actual matching input CRS. Missing/mismatched CRS is rejected
before output writing, rather than guessed or silently overridden.

Strict merge retains the exact field order and original DBF definitions:
`type, lane_type, center, Id, source, target, Label, LaneNumber, LaneWidth,
RAppend`. Lowercase `id` takes precedence over `Id`; missing fields become
null/blank according to the original PyShp field types. IDs are not renumbered.

Arrow fusion scans `.shp` files in sorted filename order. Every scanned input
must have the same declared CRS as the base. Read errors propagate instead of
silently producing an incomplete map. It selects case-insensitive `type=arrow`,
normalizes that value to lowercase, coalesces case variants of ID columns in
their original column precedence, fills missing IDs with zero, and casts them
to int32. Other fields retain the original whitelist. Inputs must be compatible
with one Shapefile geometry type. No ID deduplication or overflow correction is
added. If no arrows are found, the original no-output behavior is retained.

### Attributes

For maps exported by `construction/generate_map.py`, use
`prepare_lane_attributes.py --input lane_map.shp --output width_input.shp
--angle-threshold 10` before the width variant. The angle threshold is required
and may be 10 or 60 degrees. The adapter writes the corresponding numeric input
gate in `Label`; the width calculation replaces that gate with its computed
label. Do not reuse the resulting label as the input angle policy.

The adapter accepts only pure lane, single-part 2D linework in a declared
geographic CRS or EPSG:32650. It preserves coordinates and attributes, saves
the original `id` as `orig_id`, and creates unique `Id` values for feature scans.
Existing `Label`, `LaneNumber` and `LaneWidth` are preserved under `orig_label`,
`orig_lnum` and `orig_width`; conflicting field names are rejected. Missing
`source`/`target` remain null. These scan IDs are not graph-node identifiers.
This adapter is NOT intended for the separate label variant's numeric skip gate.

Both variants preserve field order, calculations, thresholds, and raw value
comparisons. Validation requires 2D polylines with at least two points, at least
seven fields, `id`/`Id` at index 3, and existing `LaneNumber` and `Label` fields.

Width calculation additionally requires `type` at index 0, `center` at index 2,
and an existing `LaneWidth` field. Index 6 controls the original angle threshold.
Geographic geometry is converted to EPSG:32650; projected geometry must already
be EPSG:32650. Arrow centers are always interpreted as WGS84
`longitude,latitude[,suffix]` and converted to UTM, independently of geometry
CRS. Do not rerun on centers already converted to UTM. Non-arrow center handling,
including original zero-return branches, is unchanged. The algorithm uses the
first two vertices. Vertical lines use an infinite slope and a horizontal
normal, avoiding division by zero without changing the geometric rule.
Zero-return branches can indicate unresolved neighbor relationships; a zero
width is not evidence of a physically zero-width lane.

The label variant requires EPSG:32650. Index 5 is the original skip gate; index
6 controls the angle threshold. The original numeric comparison is retained:
the string `"0"` is not numeric zero. In particular, strict fusion writes
`source`, `target`, and `Label` as strings. Its output is NOT automatically
compatible with the attribute variants' numeric gate assumptions. These gate
types must be checked by the caller; this package does not reorder or coerce
them. The label variant also retains the original multipart flattening behavior.

If an attribute input lacks `.prj`, supply its known CRS explicitly with
`--input-crs EPSG:32650` (or a geographic CRS for the width variant). A conflicting
existing `.prj` is rejected. The original coordinate-range guessing and forced
relabeling of arbitrary projected data are intentionally not retained.

### 3D to 2D

The original GDAL conversion preserves XY, copies other fields, recursively
extracts first/last Z, and replaces `center` with `start_z,end_z`. The new center
field is appended, not restored to index 2; this output must not be passed
directly to the width script without explicit schema preparation. CRS metadata
is preserved, not reprojected. Dropping ECEF Z does not create a projected
planar map. Actual conversion was verified separately using the existing system
Python 3.8.10 / GDAL 3.4.3: single and multipart features, XY coordinates,
endpoint heights, field order, CRS preservation, and overwrite rejection.

Output writes retain the original non-transactional behavior. A calculation or
driver failure may leave partial output sidecars; inspect them and select a new
output basename before retrying. No automatic deletion is performed.

## Verification

```sh
python -B -m unittest discover -s tests/public -p test_postprocessing.py
```

Tests cover import/help safety, synthetic fusion, CRS/overwrite rejection,
original synthetic attribute results, field order, schema adaptation, vertical
lines, and optional GDAL conversion.
Dependency-specific tests report skips when their libraries are unavailable.
With isolated PyShp 2.3.1 / Rtree 1.3.0 dependencies in `lane3d`, 19 tests ran:
18 passed and 1 GDAL test was skipped. The system-GDAL CLI smoke test described
above separately verified conversion; this does not turn skipped unittest
cases into passes or claim a single fully provisioned environment.
An existing 65-feature lane export was also adapted and processed by the width
script, retaining original IDs and segment metadata. It produced 65 valid
projected features with finite widths (24 nonzero). This is an execution check,
not an attribute-accuracy result. Original inputs were not modified.

## Source Provenance

Original paths and SHA256 hashes for the packaged sources are maintained in
[source_provenance.json](../source_provenance.json).
