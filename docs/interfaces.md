# Interfaces between the three lines

The single place where the three lines agree on data and code formats. If something here is unclear,
this file is wrong: fix it in your PR.

- **FIXED**: validated and used by code; change only with a note in the PR description.
- **FLEXIBLE**: the constraints listed are fixed; the rest is the owner's choice. Record the choice
  in the version folder's `MANIFEST.md` or in this file.

```
wayup.zip ──┬── external/pose2d ──> pose2d/<version>/ ──┬──> Track 1 (2D line)   ──┐
            │        (2D line)                          └──> Track 2 features     │
            │                                                (learning line)      ├──> events.csv ──> evaluation
            └── external/wham ──> distance3d/<version>/ ──┬──> Track 3 rule        │    (all tracks)   (learning line)
                     (3D line)                             └──> Track 2 features ──┘
```

## 1. Common conventions — FIXED

| Thing | Convention |
| --- | --- |
| Video id | `<recording>_<route>`, e.g. `p2b_orange`; 22 videos (`wayup.list_recordings` × `wayup.ROUTES`) |
| Frames | 0-based index into the processed `{route}.mp4`, 25 fps; intervals are inclusive `[start, end]` |
| Image coordinates | pixels in the processed 720×1280 (portrait) video, origin top-left |
| Limbs | `left_hand, right_hand, left_foot, right_foot`, always in this order (`holdon.events.LIMBS`) |
| Hold id | string of the hold's index in `{route}_routeAnnotation.json` (`"0"` … `"30"`); `"w"` = pushing on the wall, ground truth only |
| Splits | LOPO over 10 participants (`holdon.data.splits.lopo_folds` with `wayup.participant_of`; `p2a` + `p2b` = participant 2). A video's predictions must come from a model trained and tuned without its participant |
| Official data versions | `configs/paths.yaml` → `versions`. Read them from there; never hard-code a version folder |

## 2. Ground truth — FIXED

```python
from holdon.data import wayup
rec = wayup.load_recording("data/raw/wayup", "p1", "green", num_frames=4707)
rec.events      # list[LabeledEvent]: .event (ContactEvent, score 1.0) and .occluded ((a, b), ...)
rec.holds       # list[Hold]: hold_id, x1, y1, x2, y2, is_foothold
rec.wall        # WallReference: bolt-hole img/world coordinates and image->wall homography
```

939 events in total (481 hand, 458 foot; one unparseable row skipped). Pass `num_frames`
(from `video_info` or the pose file) so the few end frames one past the video are clipped.
Quirks: [dataset_schema.md](dataset_schema.md).

## 3. 2D pose — FIXED (2D line)

- Files: `data/interim/pose2d/<version>/<video_id>.npz`, one per video, all 22 videos.
- Format and keypoint names: [pose2d_format.md](pose2d_format.md). Read with
  `holdon.data.external.load_pose2d(path)`; take keypoints by name: `pose.track("left_wrist")`.
- `versions.pose2d` (RTMW, 133 keypoints incl. fingertips and toes) is the pose input for every track;
  `versions.pose2d_vitpose` (17 keypoints) exists only for the comparison with the paper.
- Scores are on a model-specific scale (RTMW about 0–10, ViTPose 0–1): thresholds go in configs, per model.

## 4. Contact events — FIXED (every track)

- Format: [event_format.md](event_format.md), code `holdon.events.ContactEvent`,
  `write_events` / `read_events`.
- One run of a track writes `results/<track>/<run name>/events.csv` with predictions for **all 22 videos**,
  plus the `config.yaml` it used. Example: `results/track1/rtmw-x-384_2026-11-18_wrist_ankle/`.
- Hold ids must be ids of the video's route (never `"w"`). Higher `score` = more confident.
- To turn distances or per-frame scores into events, use `holdon.models.rules.events_from_distances`
  (distance ≤ threshold) or `holdon.models.decoding.decode_events` (boolean mask), so all tracks share
  one minimum-duration and gap-merging rule (`min_duration_s`, `merge_gap_frames` in each track's config).

## 5. Evaluation — FIXED behaviour, FLEXIBLE code layout (learning line)

The protocol is [evaluation_protocol.md](evaluation_protocol.md). Implementation: `holdon.evaluation`
(`matching.py`: `tiou`, `match_events`; `metrics.py`: `video_counts`, `macro_metrics`, `evaluate`;
`grouped.py`: breakdowns; `report.py`: Markdown tables), tested in `tests/evaluation/`. The scoring code
was moved unchanged from `holdon/models/track1_tuning.py` (2026-11-21); `scripts/tune_track1.py` uses it,
and `scripts/evaluate.py` on the Track 1 `events_lopo.csv` reproduces the tune_track1 numbers exactly.
Entry points:

```python
from holdon.evaluation.metrics import evaluate
table = evaluate(pred_by_video, gt_by_video, thresholds=[0.0, 0.3, 0.5], limbs=None)
# pred_by_video: {video_id: [ContactEvent]}, gt_by_video: {video_id: [ContactEvent or LabeledEvent]}
# -> {0.0: {"precision", "recall", "f1", "mean_tiou"}, 0.3: {...}, 0.5: {...}}  (macro over videos)
```

```bash
python3 scripts/evaluate.py results/track1/rtmw-x-384_2026-11-18_wrist_ankle/events.csv
# prints the table and writes metrics.md next to events.csv
```

`metrics.md` contains all / hands / feet, each route, occluded vs visible frame recall and the
climber-experience groups (`configs/eval.yaml` → `experience`). `--out` changes the file name (Track 1
run folders already hold a `metrics.md` from `tune_track1.py`). Extra arguments may be added; the call
above must keep working.

## 6. 3D limb-to-hold distances — FIXED format, FLEXIBLE method (3D line)

The only Track 3 output other lines read. Raw WHAM files stay inside the 3D line (`data/interim/wham/`,
keys documented in `external/wham/README.md`).

- Files: `data/processed/distance3d/<name>_<YYYY-MM-DD>/<video_id>.npz` + `MANIFEST.md`; when ready, set
  `versions.distance3d` in `configs/paths.yaml`. Read with `holdon.data.external.load_distance3d`.

| Key | dtype, shape | Content |
| --- | --- | --- |
| `dist` | float32 (T, 4, H) | distance in **metres** from limb `l` to hold `h` in frame `t`; NaN = unknown; ≥ 0 |
| `limbs` | str (4,) | exactly `holdon.events.LIMBS` |
| `hold_ids` | str (H,) | every hold of the route, in `routeAnnotation.json` order (`"0"` … `"H-1"`) |
| `video_id` | str | e.g. `p1_green` |
| `method` | str | `"3d"` (WHAM body + wall pose) or `"2.5d"` (fallback: distance in the wall plane) |
| `format_version` | int32 | `1` |

- **Fixed:** T equals the number of frames of the processed video (row `t` = frame `t`, same as the
  pose files); metres (the bolt-hole wall coordinates are very likely cm, see dataset_schema.md: check
  against the hold spacing, then divide by 100); 0 when touching or inside.
- **Flexible (record in MANIFEST):** which body point represents each limb (e.g. SMPL wrist/hand joint or
  the closest hand vertices), how a hold is modelled (its box mapped to the wall plane via the homography
  is the default), the focal-length estimate, any smoothing.
- Track 3's own events: `events_from_distances(d.limb(limb), d.hold_ids, threshold=cfg["distance_threshold_m"], ...)`
  with `configs/track3.yaml`.

## 7. Track 2 features — FLEXIBLE (learning line)

Internal to the learning line. Constraints: build features only from the pose files (section 3), the
distance files (section 6) and hold boxes; one row per video frame (and limb–hold pair if needed),
aligned by frame index; store versioned tables under `data/processed/features/<name>_<date>/`.
Reusable 2D cues: `holdon.features.overlap2d` (keypoint-to-box distances), `holdon.preprocessing.smoothing`.

Choice for v1 (`holdon.features.build`, `configs/track2.yaml`): one row per candidate (frame, limb, hold),
where the candidates of a limb in a frame are the holds within `candidates.radius_px` (80 px) that are also
among its `candidates.max_per_limb` (3) nearest holds. Features are computed on the fly (about 4 s for all
22 videos), so no table is stored yet; the candidate coverage of the ground truth is written to
`results/track2/<run>/candidates.md`.

## 8. Versioned outputs — FIXED

Every derived dataset (pose, WHAM, distances, features, checkpoints) lives in a new folder
`<name>_<YYYY-MM-DD>/` with a `MANIFEST.md` ([template](drive_manifest_template.md)), on the team Drive
under the same path as in `data/`. Never overwrite a version that is on the Drive; make a new one.
