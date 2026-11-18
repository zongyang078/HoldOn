# 2D pose file format (`format_version` 2)

Written by `external/pose2d/run_pose2d.py`, read by `holdon.data.external.load_pose2d`.
One compressed NumPy archive per video: `data/interim/pose2d/<model>_<YYYY-MM-DD>/<video_id>.npz`,
where `video_id` is `<recording>_<route>`, e.g. `p2b_orange`.

| Key | dtype, shape | Content |
| --- | --- | --- |
| `keypoints` | float32 (T, K, 2) | pixel `x, y` in the processed 720×1280 video |
| `scores` | float32 (T, K) | per-keypoint confidence (scale depends on the model, see below) |
| `bbox` | float32 (T, 4) | climber box `x1, y1, x2, y2` given to the pose model |
| `num_persons` | int16 (T,) | person boxes detected in the frame (> 1: another person visible) |
| `all_boxes` | float32 (T, 8, 4) | every detected person box (up to 8), NaN-padded; allows re-selecting the climber offline |
| `route_region` | float32 (4,) | region the climber box was restricted to (NaN if `--no-route-filter`) |
| `keypoint_set` | str | `coco17` (K = 17) or `coco_wholebody133` (K = 133) |
| `keypoint_names` | str (K,) | name of each keypoint column |
| `video_id`, `model`, `detector` | str | identifiers |
| `fps`, `width`, `height` | scalar | from the video header |
| `source_num_frames` | int32 | frame count in the video header |
| `format_version` | int32 | `2` (v1 files had no `all_boxes` / `route_region` and could track the belayer; do not use them) |

- Row `t` is frame `t` (0-based), matching the frame numbers in `*_holdUsage.csv`.
  T normally equals `source_num_frames`; it is smaller only if the end of a video could not be decoded.
- Frames with no detected person: `keypoints` and `bbox` are NaN, `scores` are 0.
- **Climber selection:** a detected person is a candidate only if at least half of its box lies inside
  the route region (bounding box of all hold boxes in `*_routeAnnotation.json`, grown by 80 px). Among
  candidates the tracker follows the previous frame's box (IoU, then nearest centre). Without this filter
  (v1) the tracker jumped to the belayer when the climber, small and high on the wall, was missed for a
  frame: in p1_green 71% of contact frames had the limb near the labeled hold, against 96–99% in clean videos.
  Checked on v1 output: the filter keeps 100% of frames where the climber was tracked and removes 91%
  of the frames where another person was tracked.
- Coordinates are raw model output: no smoothing and no confidence filtering (that is `holdon.preprocessing`).
- Keypoints are always selected by **name**, never by hard-coded index (`Pose2D.track("left_wrist")`).

## Keypoint names

`coco17`: `nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow,
right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle`.

`coco_wholebody133` (MMPose order and names): the 17 COCO body points, then the feet
`left_big_toe, left_small_toe, left_heel, right_big_toe, right_small_toe, right_heel`, then `face_0 … face_67`,
then each hand (`left_*` first, then `right_*`) as `<side>_hand_root` followed by
`<side>_<finger><1-4>` for `thumb, forefinger, middle_finger, ring_finger, pinky_finger`
(`4` is the fingertip, e.g. `right_forefinger4`).

## Score scales

- `vitpose*`: heatmap maxima, roughly 0–1.
- `rtmw*`: SimCC confidences, **not** probabilities; typical values on The Way Up are 3–7 for visible
  points (measured on p1_green). Confidence thresholds must therefore be set per model in the config,
  never shared between models.
