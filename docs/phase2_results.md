# Phase 2 — results, facts and report plan

Source material for the Phase 2 progress report and the 20-minute presentation (due Nov 23, 2026).
Whoever writes the report or slides: take numbers, captions and claims **from this file
only**; each one names the file it comes from. Sections marked _pending_ are filled in by their owner.

Course requirements for the report and presentation: overview of the implemented solution; system
architecture and implementation approach; technologies, tools and libraries; initial testing results with
screenshots or demonstrations; challenges and planned improvements; contributions of each member.
No length or structure is prescribed; the presentation slot is 20 minutes per group.

## 1. Data and versions

| Item | Value | Source |
| --- | --- | --- |
| Dataset | The Way Up (Maschek & Schedl, CVPRW 2025), 22 videos = 11 recordings × 2 routes, 720×1280, 25 fps | `docs/dataset_schema.md` |
| Ground truth | 939 contact events (481 hand, 458 foot); 31 are wall pushes (`w`, no hold) | `docs/evaluation_protocol.md` |
| Splits | leave-one-participant-out, 10 folds (`p2a` + `p2b` = one climber) | `docs/interfaces.md` §1 |
| 2D pose version | `rtmw-x-384_2026-11-18` (official), `vitpose-plus-huge_2026-11-18` (paper comparison) | `configs/paths.yaml` |

## 2. 2D pose pipeline (2D line)

- Per frame: YOLOX-m person detection → climber box = a detected person with ≥ 50% of the box inside the
  route region (bounding box of the route's hold boxes + 80 px), tracked by IoU → pose model on that box.
  Models: **RTMW-x** (133 whole-body keypoints incl. fingertips and toes, ONNX via rtmlib) and
  **ViTPose++-H** (17 COCO keypoints, HuggingFace transformers).
- Ran on a GCP VM (n1-standard-4 + NVIDIA T4): RTMW 14 fps, ViTPose++-H 5 fps; about 2 h 45 min for
  all 22 videos (38,692 frames) and both models. (`external/pose2d/README.md`)
- Validation of the RTMW output (`data/interim/pose2d/rtmw-x-384_2026-11-18/MANIFEST.md`):
  - frame count matches the video for all 22 videos;
  - in labeled contact frames the wrist / ankle keypoint is within 40 px of the labeled hold box in
    **95.7%** of the frames where the climber was detected (first version without the route filter: 89.3%);
  - the climber is detected in 92.2% of labeled contact frames.
- ViTPose output (`vitpose-plus-huge_2026-11-18/MANIFEST.md`): same boxes as RTMW in all 22 videos;
  wrist / ankle within 40 px of the labeled hold in **97.1%** of detected contact frames.

## 3. Track 1 method

- A limb touches a hold if any of its keypoints is within `distance_px` of the hold box; per frame the
  limb is assigned to the nearest such hold only. Runs separated by ≤ `merge_gap_frames` are merged;
  contacts shorter than `min_duration_s` are dropped. Keypoints below a confidence threshold are treated
  as missing; gaps ≤ 5 frames are interpolated; median smoothing. (`src/holdon/models/rules.py`)
- Keypoint sets: **wrist/ankle** (as in The Way Up) and **fingertip/toe** (5 fingertips per hand;
  big toe, small toe, heel per foot).
- Thresholds tuned with LOPO: for each fold the best of 480 combinations on the 9 training participants
  (objective: mean F1 over the three tIoU thresholds), applied to the held-out participant. The grid
  (`configs/track1.yaml`) was widened twice because the first runs chose values at its edges.
- Chosen values (`results/tables/track1_*_chosen_params.csv`):

| Pose, keypoints | distance_px | min_duration_s | merge_gap_frames | min_score |
| --- | --- | --- | --- | --- |
| RTMW fingertip/toe | 20 (9 of 10 folds) | **2.0 (10 of 10)** | 40 (9 of 10) | 1.5 (8 of 10) |
| RTMW wrist/ankle | 45 (9 of 10) | 2.0 (9 of 10) | 25 (8 of 10) | 0.25 (9 of 10) |
| ViTPose wrist/ankle | 30 (10 of 10) | 2.0 (10 of 10) | 40 (10 of 10) | 0.05 (10 of 10; ViTPose scale 0–1) |

## 4. Track 1 results

Macro average over 22 videos, held-out predictions only. P / R / **F1** at tIoU > 0, ≥ 0.3, ≥ 0.5.
(`results/tables/track1_*.md`)

| Setting | tIoU > 0 | tIoU ≥ 0.3 | tIoU ≥ 0.5 |
| --- | --- | --- | --- |
| RTMW fingertip/toe, all limbs | 0.890 / 0.848 / **0.864** | 0.869 / 0.827 / **0.844** | 0.814 / 0.773 / **0.790** |
| — hands | 0.834 / 0.832 / **0.828** | 0.799 / 0.796 / **0.792** | 0.726 / 0.721 / **0.719** |
| — feet | 0.959 / 0.868 / **0.908** | 0.956 / 0.865 / **0.905** | 0.923 / 0.835 / **0.874** |
| RTMW wrist/ankle, all limbs | 0.757 / 0.755 / **0.752** | 0.737 / 0.734 / **0.731** | 0.654 / 0.649 / **0.648** |
| — hands | 0.856 / 0.848 / **0.847** | 0.835 / 0.826 / **0.825** | 0.762 / 0.745 / **0.749** |
| — feet | 0.656 / 0.662 / **0.652** | 0.638 / 0.644 / **0.634** | 0.550 / 0.554 / **0.546** |
| Untuned rule (0.5 s, inside box), fingertip/toe | 0.548 / 0.761 / **0.631** | 0.370 / 0.512 / **0.426** | 0.219 / 0.306 / **0.254** |
| ViTPose++-H wrist/ankle (paper setting) | 0.799 / 0.796 / **0.795** | 0.774 / 0.771 / **0.770** | 0.711 / 0.706 / **0.706** |
| — hands | 0.881 / 0.870 / **0.872** | 0.864 / 0.853 / **0.856** | 0.809 / 0.797 / **0.800** |
| — feet | 0.717 / 0.723 / **0.715** | 0.685 / 0.690 / **0.683** | 0.617 / 0.617 / **0.612** |

Findings (each follows from the table above):

1. **Toe keypoints fix the feet:** foot F1 @>0 rises from 0.652 (ankle) to 0.908 (toes). The ankle lies
   outside the foothold box when the climber stands on the toes.
2. **Hands are the hard part:** the best hand F1 is 0.872 (ViTPose wrist), below the best feet (0.908, RTMW
   toes). For hands the wrist works better than the fingertips (RTMW 0.847 vs 0.828), and ViTPose's wrist
   better than RTMW's (0.872 vs 0.847). Combining the best hand and foot keypoints is the next experiment.
3. **Pose model:** with the same boxes and wrist/ankle keypoints, ViTPose++-H beats RTMW-x
   (F1 0.795 vs 0.752 @>0; wrist/ankle within 40 px of the labeled hold 97.1% vs 95.7%), at about a third
   of the speed. RTMW stays the main pose input because only it has fingertips and toes.
4. **The 0.5 s rule is too short here:** every fold chose a 2.0 s minimum. Longer minimum durations remove
   brief false contacts (hands moving past holds); the untuned 0.5 s rule reaches only F1 0.631 / 0.426 / 0.254.
   Cost: 137 of 939 labeled contacts (14.6%, 66 of them hands) last under 2 s and cannot be found with this setting.
5. **Interval quality:** mean tIoU of matched events is 0.836 (fingertip/toe), so most detected contacts
   also have accurate start and end frames.

**Comparison with the paper:** The Way Up reports about 86.6% (tIoU > 0) and 69% (tIoU > 0.5) *accuracy*
for its ViTPose baseline; our closest setting (ViTPose++-H, wrist/ankle, LOPO-tuned) reaches F1 0.795
(tIoU > 0) and 0.706 (tIoU ≥ 0.5). Their accuracy is defined differently from our F1, so present the two
side by side as context only, never as a like-for-like comparison (`docs/evaluation_protocol.md`).

## 5. Figures (`results/figures/`, made by `scripts/make_track1_figures.py`)

| File | Caption to use |
| --- | --- |
| `track1_overlay_p8_orange.jpg` | Track 1 on p8_orange at 18.8 s. Orange: labeled contacts; magenta: Track 1 contacts; red: fingertip and toe keypoints; green / blue: hand / foot hold boxes. All four limbs are detected on the correct holds (both hands matched on hold 19, feet on holds 11 and 13). |
| `track1_timeline_p8_orange.png` | Labeled (orange) vs Track 1 (magenta) contacts per limb over the whole climb of p8_orange; numbers are hold ids. Most contacts match; errors are short extra contacts between holds and contacts split in two. |
| `track1_false_positive.jpg` | Typical false positive: p2a_green at 21.6 s, the right hand's fingertips lie inside hold 11's box for 4.2 s while the hand is not on the hold (no labeled contact). 2D overlap cannot tell whether the hand is on the hold or in front of it, which motivates the temporal (Track 2) and 3D (Track 3) cues. |
| `pose_selection_fix.jpg` | Climber selection before and after the fix: without the route-region filter the tracker jumped to the belayer (left, red box) when the climber was missed for a frame high on the wall; with it the climber stays tracked (right, green box). |

## 6. Challenges (2D line)

| Challenge | What happened | Resolution / status |
| --- | --- | --- |
| Tracking the wrong person | When the climber was small and missed for a frame, the tracker switched to the belayer (26% of frames show more than one person). Wrist near labeled hold dropped to 71% in three videos. | Route-region filter; re-ran all videos; 95.7% overall. |
| Climber not detected high on the wall | 20–30% of labeled contact frames have no climber box in p2b_green (71.1% detected), p2b_orange (71.2%), p7_green (80.8%). | Open: larger detector input or tracking across missed frames (Phase 3). |
| Hand false positives | A hand in front of a hold looks like a contact in 2D (figure `track1_false_positive.jpg`). Hand precision 0.834 @>0. | Long minimum duration helps; Tracks 2 and 3 target the rest. |
| Occlusion | Hands are > 50% occluded in about half of their contact frames (dataset paper); occluded fingertip keypoints drift. | Open; occlusion breakdown comes with the evaluation module. |
| Score scales | RTMW confidences are 0–10, not probabilities. | Thresholds configured per model. |
| GPU access | New GCP project had 0 GPU quota; then L4 and T4 GPUs were out of stock in every US zone. | Quota raised (approved in minutes); automatic retries got a T4 about 30 min later; a Colab notebook is the fallback. |
| Tuning grid | First grid chose values at its edges (results underestimated). | Grid widened twice until the chosen values were inside. |

## 7. Planned improvements (2D line, Phase 3)

- Hybrid keypoints: the best hand keypoint (ViTPose wrist) with the best foot keypoints (RTMW toes).
- Fewer missed climber frames high on the wall (detector input size, tracking through gaps).
- Kilter board clips: recording, labeling script, LED-colour hold detection (HSV), cross-dataset test.
- Faster pose export (fp16, batching, reuse of detections; `external/pose2d/README.md`).

## 8. Technologies, tools and libraries

Python 3.14, OpenCV 5 (video I/O, drawing), NumPy, matplotlib, PyYAML, scikit-learn / PyTorch (Track 2);
rtmlib 0.0.16 + ONNX Runtime GPU 1.32 (YOLOX-m detector, RTMW-x), HuggingFace transformers 5.23 + PyTorch
2.15 (ViTPose++-H); Google Cloud (n1-standard-4, NVIDIA T4) and a Colab notebook for pose inference;
pytest (58 unit tests, CPU only); GitHub (one branch per line, pull requests); Google Drive (data, versioned outputs).
WHAM + SMPL for Track 3 (3D line).

## 9. Learning line and 3D line — _pending_

To be filled in by Yen-Wei Lin (evaluation module, Track 2 v1) and Jinghong Yang (WHAM, wall pose,
3D distances): what runs, first numbers or figures, challenges, next steps.

## 10. Contributions (by role)

| Member | Line | Phase 2 work |
| --- | --- | --- |
| Zongyang Li | 2D line | Data and Drive distribution; 2D pose export (RTMW, ViTPose++) with climber selection, GCP runs and validation; Track 1 baseline and fingertip/toe variant with LOPO tuning; reference scoring; interfaces and documentation; figures |
| Yen-Wei Lin | Learning line | Evaluation module; Track 2 v1 (2D overlap + stillness) — _details pending_ |
| Jinghong Yang | 3D line | WHAM inference; wall pose from the homography; per-frame 3D distances — _details pending_ |

## 11. Report outline (suggested, 3–5 pages)

1. **Overview** — problem, research question (2D methods over-report contacts), three tracks, what Phase 2 delivers.
2. **System architecture** — data flow (video → pose / WHAM → events → evaluation), interfaces between lines
   (diagram in `docs/interfaces.md`), repository layout.
3. **Implementation** — data and splits (§1); 2D pose pipeline (§2); Track 1 (§3); Track 2 and Track 3 (§9).
4. **Tools and libraries** (§8).
5. **Initial results** — Track 1 table and findings (§4), figures (§5), Track 2 / 3 status (§9).
6. **Challenges and planned improvements** (§6, §7, plus the other lines).
7. **Contributions** (§10).

## 12. Slide outline (suggested, 20 min: about 17 min talk + 3 min questions)

Each track gets the same space (2 slides); shared topics come before and after the tracks.

| # | Slide | Min | Content | Material |
| --- | --- | --- | --- | --- |
| 1 | Title | 0.5 | project name, team, one-line problem statement | — |
| 2 | Problem and research question | 1 | which limb touches which hold and when; 2D methods over-report contacts (hand in front of a hold); question: how far do temporal and 3D cues reduce false contacts | §11.1, `track1_false_positive.jpg` |
| 3 | Approach | 1 | the three tracks in one line each; same data, splits and evaluation for all | §11.1 |
| 4 | System architecture and tools | 1.5 | data-flow diagram (video → pose / WHAM → events → evaluation); interfaces between modules; main libraries | `docs/interfaces.md` diagram, §8 |
| 5 | Data and evaluation | 1.5 | The Way Up (22 videos, 939 contacts, occlusion); LOPO splits; event matching by limb + hold, tIoU, P / R / F1 | §1, `docs/evaluation_protocol.md` |
| 6 | 2D pose pipeline | 1.5 | detector → climber selection → RTMW / ViTPose; GCP run; validation numbers; selection fix | §2, `pose_selection_fix.jpg` |
| 7 | Track 1: method and results | 1.5 | rule (keypoint near hold, minimum duration), LOPO tuning, results table | §3, §4 |
| 8 | Track 1: examples and errors | 1 | overlay and timeline; findings (toes for feet, 2 s minimum, hand false positives) | §4 findings, §5 |
| 9 | Track 2: evaluation module and features | 1.5 | evaluation module status; per-frame features (2D overlap, stillness) | §9 |
| 10 | Track 2: first results | 1.5 | classifier v1 and first numbers or examples | §9 |
| 11 | Track 3: WHAM and wall pose | 1.5 | WHAM body on the climbing videos; wall plane from the bolt-hole homography | §9 |
| 12 | Track 3: 3D distances | 1 | first limb-to-hold distance curves vs labeled contacts | §9 |
| 13 | Challenges | 1 | one row per line: what went wrong, what was done | §6 + other lines |
| 14 | Next steps (Phase 3) | 1 | per line: planned improvements; three-track comparison, ablation, Kilter demo | §7 + other lines |
| 15 | Contributions | 0.5 | one line per member | §10 |
