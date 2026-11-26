# Semantic Source Tools

Tools for semantic image/LiDAR fusion and arrow-marking inspection.

## Included Sources

| File | Scope |
| --- | --- |
| `myConcateTemp.py` | Actual extracted-image/LiDAR/TUM fusion, including arrow-mask recovery and dense arrow sampling |
| `arrow_postprocess.py` | Actual image-mask geometry, vanishing-point and region-growth helpers |
| `predict.py` | Original colormaps and optional legacy prediction factories |
| `arrow_vectorized.py` | Research GUI for yellow-point DBSCAN clusters and axis-aligned bounding boxes; no map export |
| `study_calibration.json` | Original matrix values as an explicitly selected study setting, never an automatic default |
| `configs/segformer_whu_inference.py` | Self-contained SegFormer inference architecture and preprocessing settings |
| `LICENSE` | Source project's MIT notice, copyright 2022 ebhrz |

No weights, raw dataset, car mask, Detectron2, Mask2Former, or MMSeg source
tree is vendored. Arrow rectangles are exported by
`construction/generate_map.py --with-arrows`, not by these semantic preparation
tools. Junction-graph and lane-width exports are not implemented here.
In particular, the name `arrow_vectorized`
does not imply a paper arrow exporter.

## Packaging Changes

- No model loading, dataset reading or GUI launch at module import. NumPy is
  still required for the original numerical helpers and colormaps.
- Model configuration, weights, car mask, calibration, data root, trajectory,
  output and start-time gate are explicit CLI inputs. Select the start time
  for the actual sequence.
- `extract_arrows_from_segmentation(..., car_mask=mask)` now receives the mask
  from its caller instead of rereading `car_mask.npy` from the current directory.
- `pypcd4.PointCloud.from_path(path).pc_data` replaces the `sys.modules`
  alias. The installed pypcd4 1.4.3 source confirms this API. Six original PCD
  fields remain required: x, y, z, intensity, timestamp, ring. The unused
  `pclpy` dependency is removed.
- Optional imports happen at use sites. The default CLI uses
  `mmseg.apis.MMSegInferencer` directly; it does not call either legacy factory
  in `predict.py`. Detectron2/Mask2Former are needed only when explicitly
  calling `get_predict_func_detectron`. The external original `imseg` layout
  and an importable `mask2former` package are supported; neither is bundled.
- Existing output/log paths are refused, empty results fail, and the PCD
  writer's return value is checked. These are I/O checks, not fitting changes.

The eight original fusion helper bodies are unchanged. Queue synchronization,
the 0.05-second default tolerance, last-point-wins image projection, y >= 1
pixel filter, arrow sampling (0.5 in the fusion loop), interpolation, 2-decimal
deduplication and semantic colors are preserved. NumPy's random sampling is
not newly seeded. The original `np.matrix` use is retained and may warn.

## Inputs And Coordinates

`--data-root` must contain `image/`, `lidar/`, `image_list.txt` and
`lidar_list.txt`. Lists retain the original contract: ordered timestamp-named
`.png`/`.pcd` basenames in seconds, one per line with a terminating newline.
The supplied pose file is TUM `timestamp x y z qx qy qz qw`, with matching
timestamps and coordinate frames. No timestamp-unit conversion is added.

`--calibration` reads JSON keys `intrinsic` (3x3) and `extrinsic_r2c` (4x4,
LiDAR to camera). Optional `extrinsic_r2b` is retained as metadata; the
original pose transform applies TUM directly to LiDAR-frame points and does
not multiply this body extrinsic. Sensor calibration and pose conventions
must therefore actually match; they are not inferred or corrected here.

`study_calibration.json` records the original values but is appropriate only
for that sensor/image/pose setup. For other data, provide a different JSON
with the correct matrices. This tool does not apply new undistortion or CRS
conversion. The output cloud remains in the supplied poses' world frame.

The original model must produce an HxW label array with 1=solid, 2=dashed,
3=arrow and 5=stop-line. Their PCD colors remain green, red, yellow and blue;
other labels are black. A generic segmentation checkpoint with different
classes is not interchangeable. The `.npy` car mask must have the same HxW
shape as the predictions; no resizing or class remapping is introduced.

## Commands

Run from the repository root. Help needs NumPy but not CUDA, MMSeg, Open3D,
OpenCV, SciPy or scikit-learn:

```bash
python -m construction.semantic.myConcateTemp --help
python -m construction.semantic.arrow_vectorized --help
```

Example only: set these explicit paths and the start time for your data.
Use the bundled config for the matching WHU SegFormer checkpoint. A different
checkpoint requires its own compatible configuration.

```bash
python -m construction.semantic.myConcateTemp \
  --data-root "$DATA_ROOT" --pose-path "$POSE_TUM" \
  --calibration "$CALIBRATION_JSON" \
  --model-config construction/semantic/configs/segformer_whu_inference.py \
  --weights "$WEIGHTS" \
  --car-mask "$CAR_MASK" --start-time "$START_SECONDS" \
  --device cuda --max-frames 5 \
  --output-pcd results/semantic_trial/semantic.pcd \
  --log-file results/semantic_trial/generation.log
```

`--max-frames 0` retains the original all-aligned-frames behavior. Execution is
headless by default; `--vis` enables the original dynamic-cloud GUI. The tool
does not write a new TUM file: the explicit source poses remain the trajectory
input for downstream segmentation. It does not call lane fitting itself.

### Model Configuration

The bundled configuration contains the model, image preprocessing, sliding-window
inference and inferencer pipeline. It has no `_base_` imports or training-data
paths. It uses 19 output channels to match the existing checkpoint, even though
fusion consumes only the marking classes listed above. Do not change the head
to six channels or substitute arbitrary Cityscapes weights.

Backbone-only initialization is disabled because `--weights` must supply the
full trained model. No pretrained backbone is downloaded. This is an inference
configuration, not a training recipe. The checkpoint's display-class metadata
does not establish the marking-class mapping; fusion retains its explicit
numeric label convention.

Research inspection, on a machine with a display:

```bash
python -m construction.semantic.arrow_vectorized --input "$SEMANTIC_PCD"
```

This opens Open3D windows and writes no vector map. Original DBSCAN settings
remain `eps=1.2`, `min_points=50`.

## Dependencies And Verification

`requirements-semantic.txt` lists the numerical, image and PCD dependencies.
Inference additionally needs a mutually compatible PyTorch/MMCV-with-ops/
MMEngine/MMSegmentation installation and trusted, matching model files.
Python model configs can execute code; only use trusted configs/checkpoints.
These external assets have separate licenses; the project MIT notice does
not establish permission to redistribute them.

Reference environment metadata: Python 3.8.20, NumPy 1.24.4, SciPy
1.10.1, scikit-learn 1.3.2, OpenCV distribution 4.1.2.30, Open3D 0.19.0,
pypcd4 1.4.3, torch 1.9.1+cu111, MMCV 2.1.0, MMEngine 0.10.7 and
MMSegmentation 1.2.2. This is an observed environment, not a tested portable
lockfile or an ABI-compatibility guarantee. Legacy prediction-factory return
conventions are preserved and remain backend/version-specific.
The sensor-frame CPU check used the Conda environment's native library directory
on the loader path. Mixing system and Conda C++ libraries can cause
`GLIBCXX_3.4.29` import errors; use a consistent activated inference environment.

```bash
python -B -m unittest discover -s tests/public -p test_semantic_source.py -v
```

Tests cover import/help isolation, explicit-input parsing, calibration,
synthetic numerical helpers, car-mask recovery and tiny ASCII PCD I/O.
The configuration was also checked against the checkpoint's embedded model
settings, loaded with the existing weights on CPU, and used for one real-image
inference producing a 720x1280 label array. This checks execution and output
shape, not segmentation accuracy or historical reconstruction equivalence.
One synchronized image/LiDAR/pose frame also completed the fusion entry with
`--device cpu --max-frames 1`; its PCD was read back with 5,149 finite points
and aligned RGB colors. A one-frame run is not a full-road reconstruction test.
