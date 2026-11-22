# WHAM (separate environment, 3D line)

WHAM has heavy dependencies and needs a GPU, so it runs in Colab inside WHAM's own environment.
The main `holdon` package never imports WHAM.

## What the 3D line must deliver

The **distance files** specified in [docs/interfaces.md](../../docs/interfaces.md) section 6:
`data/processed/distance3d/<name>_<YYYY-MM-DD>/<video_id>.npz`, all 22 videos, checked with
`holdon.data.external.load_distance3d`, plus `MANIFEST.md`; then set `versions.distance3d` in
`configs/paths.yaml`. Everything below is internal to the 3D line.

## Why not WHAM's `demo.py`

WHAM's demo detects and tracks people itself (YOLOv8 + ViTPose). On The Way Up the climber is small
when high on the wall and its tracker loses them: on `p3_orange` (967 frames) it returned three short
tracks (166, 37 and 46 frames, at most 26% of the video), and the mesh disappeared once the climber had
climbed a few metres. Where it did run, the mesh fitted the climber well.

[`run_wham.py`](run_wham.py) therefore replaces only the detection and tracking step. It takes the
climber's COCO-17 keypoints from the 2D line's ViTPose++-H pose files (`versions.pose2d_vitpose` in
`configs/paths.yaml`), which already follow the climber only (route-region filter, no belayer) and use
the same keypoint layout and 0–1 score scale as WHAM's ViTPose. The rest is WHAM's own code, as in
`demo.py`: boxes from the visible keypoints (×1.2, median-smoothed), HMR2 image features, the WHAM
network with flip averaging, SMPL. Static camera: no SLAM (`--estimate_local_only` equivalent), no SMPLify.

Segments: frames with ≥ 6 keypoints above 0.3 (WHAM's thresholds); gaps of up to `--max-gap` frames
(default 12, about 0.5 s) are bridged with interpolated boxes and masked keypoints; a longer gap starts a
new segment, and segments shorter than 30 frames are dropped (as in WHAM).

## Colab

Use a copy of the official WHAM demo notebook (it installs WHAM and downloads SMPL and the checkpoints;
enter your own SMPL account there, never in code). Pick a **T4 GPU** runtime, run its install and
download cells, then add these cells at the end and run them in order.

The official notebook installs conda with `condacolab` and puts WHAM in a conda env named `wham`.
After that, `python3` is conda's *base* environment, which has neither torch nor the packages of our
repository. Hence cell 2 installs our package's dependencies into base (for the dataset installer), and
cells 3 and 4 run `run_wham.py` with `source activate wham && python`.

```python
# 1. Parameters and Drive
BRANCH = "feature/3d-line"                                # use "main" after the PR is merged
DATE = "2026-11-22"                                       # version date; keep it fixed to resume
DRIVE_HOLDON = "/content/drive/MyDrive/CS5330/HoldOn"      # your path to the team folder (shortcut to My Drive if shared)
POSE_DIR = f"{DRIVE_HOLDON}/data/interim/pose2d/vitpose-plus-huge_2026-11-18"
from google.colab import drive
drive.mount("/content/drive")
import os
assert os.path.isdir(POSE_DIR), POSE_DIR
```

```python
# 2. Our repository and the dataset (local disk; reading videos over Drive is slow)
%cd /content
!rm -rf HoldOn && git clone -q --branch {BRANCH} https://github.com/zongyang078/HoldOn.git
%cd /content/HoldOn
!git log -1 --oneline
!pip3 install -q pyyaml numpy opencv-python pandas matplotlib scikit-learn   # conda base lacks them
!pip3 install -q --no-deps -e .
if not os.path.exists("data/raw/wayup/p1/green.mp4"):
    !cp "{DRIVE_HOLDON}/data/raw/wayup.zip" /content/wayup.zip
    !python3 scripts/install_data.py /content/wayup.zip
```

```python
# 3. Test on one short video, local output (about 2-3 min on a T4)
!source activate wham && python external/wham/run_wham.py --wham-root /content/WHAM --pose-dir "{POSE_DIR}" \
    --out /content/wham_test --only p3_orange --overwrite
```

The printed line must show a high `covered` share (frames with a body) and a small reprojection error
(median distance in pixels between the projected SMPL joints and the ViTPose keypoints; a few pixels is
good, tens of pixels means the body or the camera model is off). First result: `p3_orange` 96% covered
(WHAM's own tracker: at most 26%).

```python
# 4. All 22 videos, written to Drive (resumes: finished videos are skipped)
OUT = f"{DRIVE_HOLDON}/data/interim/wham"
!source activate wham && python external/wham/run_wham.py --wham-root /content/WHAM --pose-dir "{POSE_DIR}" \
    --out "{OUT}" --date {DATE}
```

Then open `MANIFEST.md` in the version folder on Drive and fill in *Created by*, *Environment* and *Notes*.

## Output: `data/interim/wham/wham-vitposebox_<YYYY-MM-DD>/<video_id>.npz`

Row `t` is frame `t` of the processed video (T = its frame count). Frames without a body are NaN
(`valid` false). 3D points are in the **camera frame in metres** (x right, y down, z forward, OpenCV
convention), with WHAM's default intrinsics.

| Key | dtype, shape | Content |
| --- | --- | --- |
| `video_id` | str | e.g. `p3_orange` |
| `format_version` | int32 | `1` |
| `num_frames`, `fps`, `width`, `height` | scalars | from the video |
| `focal_length` | float32 | WHAM default `sqrt(w² + h²)` = 1468.6 px for 720×1280 |
| `principal_point` | float32 (2,) | image centre `(w/2, h/2)` |
| `valid` | bool (T,) | WHAM output exists for the frame |
| `segment` | int16 (T,) | segment index, `-1` where not valid |
| `pose` | float32 (T, 72) | SMPL axis-angle, root in the camera frame (as `demo.py`) |
| `trans` | float32 (T, 3) | SMPL translation (as `demo.py`) |
| `betas` | float32 (T, 10) | SMPL shape |
| `joints` | float32 (T, 24, 3) | SMPL joints (`J_regressor` on the mesh) |
| `coco17` | float32 (T, 17, 3) | WHAM's COCO-17 joints (used for the reprojection check) |
| `limbs` | str (4,) | `holdon.events.LIMBS` |
| `limb_vertex_ids` | int32 (V,) | SMPL vertex ids, the four limbs concatenated in `limbs` order |
| `limb_vertex_counts` | int32 (4,) | number of vertices per limb |
| `limb_verts` | float32 (T, V, 3) | those vertices per frame |
| `reproj_err_px` | float32 (T,) | median pixel error of projected `coco17` vs ViTPose keypoints with score ≥ 0.5 |
| `keypoint_source`, `checkpoint` | str | provenance |

**Limb points.** A hand is the SMPL vertices whose dominant skinning joint is the wrist or hand joint
and that lie beyond the wrist along the elbow→wrist direction in the rest pose (palm and fingers); a foot
likewise from the ankle and foot joints, beyond the ankle along knee→ankle (sole, heel, toes). Distances
use the closest of these vertices to a hold.

The full mesh (6890 vertices) is not stored: about 390 MB per long video. SMPL model files are licensed
per person (smpl.is.tue.mpg.de): never commit or share them; the stored vertex positions are our output.

## Still to do (3D line)

The focal length is WHAM's default; the wall pose (`holdon.preprocessing.homography`) estimates its own
and the two are reconciled when the distances are computed (`holdon.features.distance3d`).
