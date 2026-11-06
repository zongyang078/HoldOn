# Dataset schema — The Way Up

Inspected on 2026-11-06 from the full Zenodo `dataset.zip` (all 22 recordings).
Loader: `holdon.data.wayup`. Alignment images: `scripts/check_alignment.py` → `results/alignment/`.

## Folder layout

```
dataset/
├── README.md                     # original dataset description
├── participant_form_data.csv     # participant demographics, see below
├── topo_green.png, topo_orange.png
└── p1, p2a, p2b, p3 … p10/       # 11 recordings, 10 participants
    ├── {green,orange}.mp4                    # USED: 720×1280, 25 fps
    ├── {green,orange}_large.mp4              # not used (4K / 50 fps)
    ├── {green,orange}_holdUsage.csv
    ├── {green,orange}_routeAnnotation.json
    └── {green,orange}_bolt_holes.json
```

- **`p2a` and `p2b` are the same climber (participant 2).** LOPO must hold out both together;
  `holdon.data.splits.lopo_folds` does this via `wayup.participant_of`.
- Video id convention used in this repo: `<recording>_<route>`, e.g. `p2b_orange`.

## Processed videos `{route}.mp4`

All 22: 720×1280 (portrait), 25.0 fps, 967–4707 frames. Container frame count equals the number
of decodable frames (checked by full decode on p4/p7/p8 green). Videos are stabilized, so hold boxes
are static; the last frames show black borders from the stabilizing warp.

## `{route}_holdUsage.csv` — contact labels

```
p1,startFrame,endFrame,framesOccluded
rh 2,47,239,"59-69, 133-195, 236-239"
lh2,73,166,"73-78, 133-166"
rf 0,100,194,
```

| Column (position) | Header | Content |
| --- | --- | --- |
| 0 | **the recording name** (`p1`, `p2a`, …), not a column name | Label `<limb><hold>` |
| 1 | `startFrame` | First contact frame, int |
| 2 | `endFrame` | Last contact frame, int |
| 3 | `framesOccluded` | Comma-separated `a-b` ranges, quoted if more than one; empty if none |

- Read columns **by position**, because header cell 0 differs per file.
- Label: two letters (`l`/`r` + `h`/`f`), then a hold id or `w` (pushing on the wall without a hold).
  Usually there is a space (`rh 2`), but 3 rows have none (`lh2`). Regex: `\s*([lr][hf])\s*(\d+|w)\s*`.
- One row does not match: `p4/green` line 27, `lf (wrong hold),854,864,`. The loader skips it with
  a warning (the hold is unknown, so it cannot be matched by limb + hold).
- **Frames are 0-based** (10 videos have `startFrame = 0`).
- In p4/p7/p8 green, `endFrame` equals the frame count (one past the last frame) for 4 rows each; the
  loader clips to `num_frames - 1` when `num_frames` is given.
- Occlusion ranges are all `a-b` (680 tokens), inclusive, and lie inside the event interval.
- Counts: 940 rows = 481 hand + 459 foot, matching the paper. After skipping the `wrong hold` row the
  loader yields 939 events (481 hand, 458 foot); 489 have at least one occluded range.
- `w` events: hands 4, feet 27.

## `{route}_routeAnnotation.json` — hold boxes

```json
{"holds": [{"topLeftCorner": {"x": 227, "y": 1030},
            "bottomRightCorner": {"x": 493, "y": 1042},
            "isFoothold": true}, ...]}
```

- Pixel coordinates in the 720×1280 processed video, origin top-left.
- **No explicit id: the hold id is the list index**, matching the numbers in `topo_*.png`
  (checked visually; max label id = number of holds − 1 in every file).
- Green: 22 holds (0–21); orange: 31 holds (0–30). Hold 0 is the wide strip at the bottom of the wall
  (start ledge), used mostly by feet.
- Boxes line up on the first and last frame of all 22 videos (`results/alignment/*.jpg`).

## `{route}_bolt_holes.json` — wall reference

| Key | Shape | Meaning |
| --- | --- | --- |
| `img_coords` | N×2 | Pixel coordinates of reference points |
| `world_coords` | N×3 | Wall coordinates, `z = 0`; x in 0–135, y in 0–510 (very likely cm) |
| `homography_mat` | 3×3 | Maps **image pixels → wall (x, y)**; absent in one file |

- 21 files: N = 4 and `homography_mat` present; applying it to `img_coords` reproduces `world_coords`.
- **`p2a/green`: N = 350 and no `homography_mat`.** The loader estimates it with
  `cv2.findHomography(..., RANSAC, 5.0)`: 350/350 inliers, median reprojection error 1.2, p95 2.2 (wall units).
- Wall y axis points up (image y points down).

## `participant_form_data.csv` — participant demographics

Shared inside `wayup.zip`. Use only for grouped analysis; never report individual values or link
them to people in videos or figures. 10 rows (`Participant ID` 1–10);
some free-text cells contain newlines, so read with a CSV parser, not line by line.

Columns: `Participant ID`, `Age`, `height (in cm)`, `gender`, `For how long have you been climbing?`,
`How often do you climb?`, `Do you prefer rope climbing or bouldering?`,
`Do you mostly climb in the gym or outdoors?`, `Is there anything else you would like to tell us?`, `notes`.

For the experience breakdown use `For how long have you been climbing?` (4 categories, no missing values).

## Kilter (self-recorded)

Same label format as The Way Up. _TODO once the labeling script exists._
