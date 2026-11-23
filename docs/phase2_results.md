# Phase 2 — results, facts and report plan

Source material for the Phase 2 progress report and the 20-minute presentation (due Nov 23, 2026).
Whoever writes the report or slides: take numbers, captions and claims **from this file
only**; each one names the file it comes from. Sections marked _pending_ are filled in by their owner.

Two separate submissions use this file:

- **Phase 2 assignment** (Implementation and Initial Testing): prototype, source code and a progress
  report covering overview, architecture, tools and libraries, initial results with screenshots,
  challenges and planned improvements, and each member's contribution. Report plan: §11.
- **Presentation, Phase 1 & 2** (300 points, slides uploaded by **Mon Nov 23, 2026, 8:59 pm**):
  20 minutes for our 3-member team, **every member presents**. Graded parts: Problem & Motivation (50),
  Existing Work & Proposed Solution (50), Implementation & Demonstration (100), Initial Testing &
  Results (50), Challenges & Team Contributions (50). Focus on what is implemented; demonstration and
  results come before background, and related work stays short. Slide plan: §12.

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

Demo videos (`results/videos/`, made by `scripts/make_overlay.py`; not in git, copy them to the Drive
folder `deliverables/phase2_progress/` for the slides). Each frame: hold boxes, labeled contact holds
(orange), Track 1 contact holds (magenta), fingertip/toe keypoints (red), and a band with labeled vs
predicted hold per limb (green row = agree, red row = disagree). Held-out (LOPO) predictions only.

| File | Content | Command |
| --- | --- | --- |
| `track1_p8_orange_0-39s.mp4` | Whole climb of p8_orange (39 s, H.264, 540×1100) | `python3 scripts/make_overlay.py --video p8_orange --scale 0.75` |
| `track1_p2a_green_17-27s.mp4` | The false positive of `track1_false_positive.jpg` in motion (right hand passes hold 11) | `python3 scripts/make_overlay.py --video p2a_green --start 17 --end 27 --scale 0.75` |

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
pytest (82 unit tests, CPU only); GitHub (one branch per line, pull requests); Google Drive (data, versioned outputs).
WHAM + SMPL for Track 3 (3D line).

## 8a. Existing work (from the Phase 1 related-work review)

| Work | What it does | Gap we address |
| --- | --- | --- |
| The Way Up (Maschek & Schedl, CVPRW 2025) | Dataset + 2D baseline: pose keypoint inside a hold box; ~86.6% accuracy at tIoU > 0, ~69% at > 0.5 (ViTPose) | 2D only; over-predicts hand contacts |
| Training-Free (arXiv 2609.30026, 2026) | Fingertip keypoints for hold contact | Still 2D; hand false positives remain; its result tables are inconsistent, so cite the idea, not the numbers |
| CIMI4D (CVPR 2023), ClimbingCap (CVPR 2025) | Accurate 3D climbing motion | Need LiDAR / extra sensors, not monocular video |
| DECO (ICCV 2023) | Body–scene contact from a single image | Not hold-specific, no timing |
| WHAM (CVPR 2024), HMR 2.0 (ICCV 2023) | Monocular 3D human motion; WHAM uses foot contact | Hand contact on walls untested |

Gap: no monocular, temporal, hold-level contact detector that compares 2D and 3D cues on the same
splits. Real-world uses (Phase 1 slide 2): competition scoring, coaching feedback, training logs; today
this is done by manual video review or force sensors built into holds, neither of which scales.

## 9. Learning line (evaluation module, Track 2 v1) and 3D line

### 9.1 Evaluation module (learning line)

- **What it is:** `holdon.evaluation` — `matching.py` (tIoU, greedy matching), `metrics.py` (per-video
  counts, macro P / R / F1, `evaluate(pred_by_video, gt_by_video, thresholds, limbs)`), `grouped.py`
  (breakdowns), `report.py` (Markdown tables); `scripts/evaluate.py <events.csv>` prints the tables and
  writes `metrics.md` next to the events file. (`docs/interfaces.md` §5)
- **Same numbers as before:** the scoring code was moved unchanged from `holdon/models/track1_tuning.py`.
  `scripts/tune_track1.py` now imports it and reproduces its earlier `metrics.md` and `events_lopo.csv`
  byte for byte; `scripts/evaluate.py` on both Track 1 `events_lopo.csv` files gives exactly the
  tune_track1 all / hands / feet rows. (`results/tables/track1_*_breakdowns.md`)
- **Breakdowns** (`docs/evaluation_protocol.md`): hands / feet (both sides filtered), route (macro over 11
  videos), occluded vs visible frame recall (ground-truth contact frames inside / outside `framesOccluded`
  covered by a same-limb, same-hold prediction, pooled over all events), climber experience (group averages
  only). Only one participant answered "less than 1 year", so that category is merged with "1-2 years"
  (`configs/eval.yaml` → `experience.min_participants: 2`) and no reported number belongs to one person.
- **Tests:** 24 more unit tests than before (old scoring tests moved; `tests/evaluation/`, `tests/features/`, `tests/models/test_svm.py`,
  `tests/training/`), all on synthetic data; 82 tests in total. 81 pass on Windows; the 82nd
  (`tests/data/test_archive.py::test_check_layout`) compares `/` paths and fails only on Windows.

### 9.2 Track 2 v1 method (learning line)

- **Candidates instead of every limb–hold pair:** per frame and limb, only holds within 80 px of the
  fingertip/toe or wrist/ankle keypoints that are also among the 3 nearest holds
  (`configs/track2.yaml` → `candidates`): 394,759 rows for 22 videos, 113,269 labelled contact.
  Label = the ground truth has this limb on this hold in this frame (`w` not counted).
- **Features (36 per row, `holdon.features.build`):** 2D overlap — fingertip/toe and wrist/ankle distance
  to the box, inside-box flags, rank of the hold, holds within 80 px, keypoint confidences, hand or foot,
  foothold or not; stillness — speed of the fingertip/toe centre and of the wrist/ankle in climber-box
  heights per frame (`holdon.features.stillness`, smoothing as in Track 1); temporal context — rolling mean
  and min of the distances and speeds over ±2 / ±6 / ±12 frames. No 3D features yet (`distance3d: false`).
- **Classifier (`holdon.models.svm`):** median imputation → StandardScaler → Nystroem RBF (300 components)
  → LinearSVC (balanced class weights), trained on a stratified sample of 60,000 rows.
- **Decoding:** per limb and frame, the candidate hold with the highest SVM score is a contact if the score
  is above a threshold; then the shared `decode_events` rule (minimum duration, gap merging); event score =
  mean SVM score of its frames.
- **Protocol (`holdon.training.train`):** outer LOPO, 10 folds. Inside each fold the 9 training
  participants are split into 3 groups; out-of-fold SVM scores of the training videos are used to choose
  threshold, `min_duration_s` and `merge_gap_frames` (objective: mean F1); the SVM is then refit on all
  training participants and applied to the held-out one, which is never used for training or tuning.
- **Chosen values** (`results/tables/track2_rtmw-x-384_2026-11-18_v1_chosen_params.csv`): threshold 0.5
  (7 of 10 folds; 0.25 in 3), `min_duration_s` 1.0 (9 of 10), `merge_gap_frames` 80 (9 of 10). The gap grid
  was widened twice because the first two runs chose its upper edge in every fold (40, then 80 frames); the
  final grid goes up to 200 frames.
- **Runtime:** about 3 min for all 10 folds on a laptop CPU (features: 4 s).

### 9.3 Track 2 v1 results (learning line)

Same format as §4: macro average over 22 videos, held-out predictions only, P / R / **F1**.
(`results/tables/track2_rtmw-x-384_2026-11-18_v1.md`; Track 1 rows copied from §4)

| Setting | tIoU > 0 | tIoU ≥ 0.3 | tIoU ≥ 0.5 |
| --- | --- | --- | --- |
| **Track 2 v1 (SVM), all limbs** | 0.947 / 0.887 / **0.915** | 0.928 / 0.869 / **0.896** | 0.883 / 0.828 / **0.853** |
| — hands | 0.923 / 0.883 / **0.899** | 0.891 / 0.854 / **0.869** | 0.829 / 0.796 / **0.809** |
| — feet | 0.973 / 0.895 / **0.931** | 0.967 / 0.890 / **0.925** | 0.945 / 0.870 / **0.904** |
| Track 1 RTMW fingertip/toe, all limbs | 0.890 / 0.848 / **0.864** | 0.869 / 0.827 / **0.844** | 0.814 / 0.773 / **0.790** |
| — hands | 0.834 / 0.832 / **0.828** | 0.799 / 0.796 / **0.792** | 0.726 / 0.721 / **0.719** |
| — feet | 0.959 / 0.868 / **0.908** | 0.956 / 0.865 / **0.905** | 0.923 / 0.835 / **0.874** |
| Track 1 RTMW wrist/ankle, hands | 0.856 / 0.848 / **0.847** | 0.835 / 0.826 / **0.825** | 0.762 / 0.745 / **0.749** |

The two earlier Track 2 runs with narrower gap grids (kept for the record, not the reported result) reached
all-limb F1 0.901 / 0.879 / 0.835 (grid up to 40 frames) and 0.912 / 0.895 / 0.859 (up to 80 frames).
(`results/track2/rtmw-x-384_2026-11-18_v1-grid1/`, `…_v1-grid2/`, local only)

Findings (each follows from the table above unless another file is named):

1. **Fewer hand false positives (the research question):** hand precision at tIoU > 0 rises from 0.834
   (Track 1 fingertips) and 0.856 (Track 1 wrists) to **0.923**, and at tIoU ≥ 0.5 from 0.726 / 0.762 to
   0.829. Pooled over the 22 videos, false hand contacts at tIoU > 0 drop from 91 (fingertips) and 76
   (wrists) to **38**, while correct hand contacts rise from 401 / 407 to 422 of 481. (pooled counts from
   `holdon.evaluation.metrics.video_counts` on the three `events_lopo.csv` files)
2. **Better overall:** all-limb F1 is 0.051 / 0.052 / 0.063 above the best Track 1 setting at the three
   thresholds. Hand F1 0.899 is also above ViTPose wrists (0.872, §4), although Track 2 uses RTMW only.
3. **Better intervals, shorter minimum:** mean tIoU of matches is 0.882 (Track 1 fingertip/toe: 0.836).
   Track 2 chose a 1.0 s minimum duration instead of 2.0 s, so only 40 of 939 labelled contacts (4.3%, 32 of
   them hands) are too short to be found, instead of 137 (14.6%). Gaps up to 80 frames (3.2 s) are merged.
4. **Occlusion** (frame recall; `results/tables/track2_rtmw-x-384_2026-11-18_v1.md` vs
   `results/tables/track1_rtmw-x-384_2026-11-18_fingertip_toe_breakdowns.md`): occluded hand frames 0.872
   (Track 1: 0.779), visible hand frames 0.885 (0.869). Occluded foot frames are worse (0.734 vs 0.845), but
   there are only 990 of them (under 2% of foot contact frames).
5. **Route and experience** (same files): F1 @>0 green 0.908 / orange 0.922 (Track 1: 0.884 / 0.845).
   Experience groups: 0.912 (less than 1 year + 1-2 years, 6 participants), 0.924 (5-10 years, 2),
   0.914 (more than 10 years, 2); no clear trend with three small groups.

### 9.4 Challenges (learning line)

| Challenge | What happened | Resolution / status |
| --- | --- | --- |
| Unreachable contacts | 8.2% of ground-truth contact frames have no candidate row for the right hold: 6.3% have no pose (climber not detected, §6) and 2.0% have the hold outside the 3 nearest within 80 px (hands 2.9%, feet 1.0%). (`results/tables/track2_rtmw-x-384_2026-11-18_v1_candidates.md`) | Reported; better climber detection (2D line) and 3D distances should recover part of it |
| Flickering frame decisions | The per-frame SVM switches on and off inside contacts; tuning compensates by merging gaps of 80 frames (3.2 s), and the first two grids were too narrow. | Grid widened until the choice lay inside it; a temporal model (GRU / 1D-CNN) should make long merges unnecessary |
| Size of the problem | Every limb × hold pair would give 4 × (number of holds) rows per frame, about 4.0 M rows for 22 videos (22 or 31 holds per route). | Candidate selection: 395 k rows, features in 4 s, full LOPO in 3 min on a CPU |
| Fair tuning | Decoding parameters must not see the test participant. | Inner 3-group split of the training participants with out-of-fold scores |
| Privacy | One experience category has a single participant. | Merged with the neighbouring category; only group averages are reported |
| Windows | `metrics.md` could not be written with the default cp1252 encoding (`≥`). | All scripts write UTF-8 |

### 9.5 Planned improvements (learning line, Phase 3)

- **3D features:** add the Track 3 limb-to-hold distances (`features.distance3d: true`) once `versions.distance3d` exists.
- **Temporal model:** GRU or 1D-CNN over feature windows (`holdon.models.temporal`, `holdon.training.windows`), same LOPO and decoding, compared with the SVM.
- **Ablation:** remove 2D overlap, stillness, temporal context and 3D cues one at a time (`features` toggles), same protocol.
- **Failure gallery:** the remaining hand false positives and missed occluded contacts as overlays and timelines (`holdon.visualization`); optical-flow stillness if keypoint speed turns out to be the bottleneck.

### 9.6 3D line — _pending_

To be filled in by Jinghong Yang (WHAM, wall pose, 3D distances): what runs, first numbers or figures,
challenges, next steps.

## 10. Contributions (by role)

| Member | Line | Phase 2 work |
| --- | --- | --- |
| Zongyang Li | 2D line | Data and Drive distribution; 2D pose export (RTMW, ViTPose++) with climber selection, GCP runs and validation; Track 1 baseline and fingertip/toe variant with LOPO tuning; reference scoring; interfaces and documentation; figures |
| Yen-Wei Lin | Learning line | Evaluation module (`holdon.evaluation`: scoring moved from Track 1 with identical numbers, `evaluate()` entry point, hands/feet, route, occlusion and experience breakdowns, `scripts/evaluate.py`); Track 2 v1 (candidate limb–hold pairs, 2D overlap + stillness + temporal-context features, Nystroem SVM, LOPO with inner-fold decoding tuning, `scripts/run_track2.py`); Track 2 results and report material (§9.1–9.5); unit tests for all new modules |
| Jinghong Yang | 3D line | WHAM inference; wall pose from the homography; per-frame 3D distances — _details pending_ |

## 11. Report outline (Phase 2 progress report, suggested 3–5 pages)

1. **Overview** — problem, research question (2D methods over-report contacts), three tracks, what Phase 2 delivers.
2. **System architecture** — data flow (video → pose / WHAM → events → evaluation), interfaces between lines
   (diagram in `docs/interfaces.md`), repository layout.
3. **Implementation** — data and splits (§1); 2D pose pipeline (§2); Track 1 (§3); Track 2 and Track 3 (§9).
4. **Tools and libraries** (§8).
5. **Initial results** — Track 1 table and findings (§4), figures (§5), Track 2 / 3 status (§9).
6. **Challenges and planned improvements** (§6, §7, plus the other lines).
7. **Contributions** (§10).

The slides stay light on purpose; the report carries the detail they leave out. Put these in the report:
the full P / R / F1 table incl. hands vs feet (§4), the chosen parameters per fold (§3), the full technique
list with versions (§8), GCP speeds and run time (§2), pose validation incl. the ViTPose numbers (§2), the
full challenge table (§6), and the paper comparison caveat (§4).

Draft: [phase2_report.md](phase2_report.md).

## 12. Slide outline (20 min, follows the five graded parts)

About 18 min of talk plus 2 min buffer or questions. Minutes follow the points: the demonstration part
(100 points) gets the most time. Each track has two slides (implementation + output, then results);
every member presents part of the talk. Upload the slides and both demo videos to the Drive folder
`deliverables/phase2_progress/`; embed the videos in the slides and keep a still image as fallback.

- **Reuse Phase 1** (`deliverables/phase1_proposal/Team06_HoldOn_Phase1_Slides.pptx` on the Drive): parts
  1 and 2 cover Phase 1 content, so condense its slides 2–3 (motivation, problem), 8–9 (existing work)
  and 7 (three tracks), and keep its visual style. Update what changed since Phase 1: pose tools are now
  rtmlib (YOLOX-m + RTMW-x) and HuggingFace ViTPose++, not MMPose / Ultralytics YOLO; the tuned minimum
  contact duration is 2.0 s, not 0.5 s; ground truth counted as 939 events.
- **Demo material per line:** the rubric asks for a working prototype on sample data with actual
  outputs (images, videos or data). Each line shows its own: Track 1 the two videos above; Track 2 e.g.
  `scripts/evaluate.py` output and a feature plot on one video; Track 3 e.g. the WHAM render on a
  climbing video and a distance curve. All on The Way Up videos (Kilter clips come in Phase 3).

| # | Part (points) | Slide | Min | Content | Material |
| --- | --- | --- | --- | --- | --- |
| 1 | — | Title | 0.3 | project name, team, one-line problem statement | — |
| 2 | 1 Problem & Motivation (50) | Hook and real-world example | 1.2 | 8 s raw clip + "which holds does the right hand touch?"; the coach's label rows; uses: competition scoring, coaching feedback, training logs; today manual review or instrumented holds | §8a, p8_orange 12–20 s and its label rows |
| 3 | 1 | Why it is hard | 1 | hands occluded about half of their contact time; a hand passing in front of a hold looks like a grasp in 2D; research question | `track1_false_positive.jpg`, §6 |
| 4 | 2 Existing Work & Solution (50) | Existing work | 0.8 | one row per work, the gap (keep short) | §8a |
| 5 | 2 | Proposed solution and CV techniques | 1.5 | three tracks in one line each, same data / splits / evaluation; techniques and why: YOLOX (climber detection), RTMW (fingertips, toes) and ViTPose (paper baseline) pose, median filtering, homography + RANSAC (wall plane), optical flow (stillness), SVM / temporal model, WHAM (3D body) | §3, §8, `docs/PROJECT_CONTEXT.md` §1 |
| 6 | 3 Implementation & Demo (100) | System workflow / architecture | 1.3 | data-flow diagram video → pose / WHAM → contact events → one evaluation module; shared event format; versioned data on the Drive | `docs/interfaces.md` diagram |
| 7 | 3 | Data and 2D pose pipeline | 1.2 | The Way Up (22 videos, 939 contacts); detect → pick the climber → 133 keypoints; one real keypoint frame; 95.7% validation (GCP details in the notes) | §1, §2 |
| 8 | 3 | Track 1: implementation | 1 | clean → measure → assign → filter; ankle vs toe image (why toes); fair LOPO tuning | §3, §4 finding 1 |
| 9 | 3 | **Demo: Track 1 on a full climb** | 1.5 | play `track1_p8_orange_0-39s.mp4`; explain the status band (green = agrees with the label) | §5 videos, `track1_overlay_p8_orange.jpg` as fallback |
| 10 | 3 | Track 2: implementation and output | 1.2 | evaluation module; per-frame features (2D overlap, stillness, context), SVM, LOPO; timeline on a sample video | §9.1, §9.2, `track2_timeline_p8_orange.png` |
| 11 | 3 | Track 3: implementation and output | 1.2 | WHAM body on a climbing video; wall plane from the homography; a limb-to-hold distance curve | §9 |
| 12 | 4 Testing & Results (50) | How we test | 0.8 | LOPO by participant; event matching by limb + hold with tIoU 0 / 0.3 / 0.5; P / R / F1 macro over videos; pose validation (95.7% near labeled hold); 82 CPU unit tests | §1, §2, `docs/evaluation_protocol.md` |
| 13 | 4 | Track 1 results | 1 | results table (fingertip/toe vs wrist/ankle vs ViTPose vs untuned); paper numbers as context only | §4 table, comparison note |
| 14 | 4 | Track 2 and Track 3 first results | 1 | Track 2 v1 vs Track 1 table (hand false contacts 91 → 38); Track 3 first output | §9.3, §9.6 |
| 15 | 4 | What works, what needs improvement | 1 | works: feet with toe keypoints (F1 0.908), accurate intervals (mean tIoU 0.836); needs work: hand false positives (play `track1_p2a_green_17-27s.mp4`), missed climber high on the wall, contacts under 2 s | §4 findings, §6 |
| 16 | 5 Challenges & Contributions (50) | Challenges and limitations | 1 | 2D line as three picture cards (wrong person, GPUs, climber too small); one box per other line | §6, `pose_selection_fix.jpg` + other lines |
| 17 | 5 | Next steps (Phase 3) | 0.7 | per line; three-track comparison, ablation, Kilter demo | §7 + other lines |
| 18 | 5 | Team contributions | 0.6 | one line per member | §10 |
