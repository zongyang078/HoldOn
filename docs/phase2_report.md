# Hold On — Phase 2 Progress Report

**Hand/Foot–Hold Contact Detection from Monocular Climbing Video**
CS 5330 Pattern Recognition and Computer Vision, Fall 2026 — Team 06: Zongyang Li, Yen-Wei Lin, Jinghong Yang
Repository: https://github.com/zongyang078/HoldOn

> Draft. All numbers come from `docs/phase2_results.md`, which names the file behind each one.
> Parts marked **[3D line — to be filled in]** are waiting for Track 3 results.

## 1. Overview

Given a climbing video and the bounding boxes of the holds, we detect **which hand or foot touches which
hold, and when**. Each output is a contact event: limb, hold id, start frame and end frame. Uses include
competition scoring, coaching feedback and training logs, which today rely on manual video review or on
force sensors built into the holds.

2D methods already find most contacts but also report false ones: a hand that passes in front of a hold
looks like a grasp in the image. Our research question is whether temporal and 3D cues reduce these
false hand contacts. We compare three tracks on the same data, splits and metrics:

1. **Track 1 — classical 2D baseline (2D line):** a pose keypoint near a hold box for long enough is a contact.
2. **Track 2 — learned per-frame classifier (learning line):** 2D overlap, stillness and temporal-context
   features → SVM (Phase 3: 3D features and a temporal model).
3. **Track 3 — 3D pipeline (3D line):** WHAM body mesh + wall pose from the bolt-hole homography →
   3D limb-to-hold distances.

Phase 2 delivers a working pipeline from video to scored contact events. It includes 2D pose for all 22
videos, Track 1 tuned with leave-one-participant-out (LOPO), a shared evaluation module, and Track 2 v1.
Track 3 status: **[3D line — to be filled in]**.

## 2. System architecture

```
wayup.zip ──┬── external/pose2d ──> pose2d/<version>/ ──┬──> Track 1 (2D line)   ──┐
            │        (2D line)                          └──> Track 2 features     │
            │                                                (learning line)      ├──> events.csv ──> evaluation
            └── external/wham ──> distance3d/<version>/ ──┬──> Track 3 rule        │    (all tracks)   (learning line)
                     (3D line)                             └──> Track 2 features ──┘
```

- **One event format for all tracks** (`src/holdon/events.py`, `docs/event_format.md`): every track writes
  `events.csv` with video, limb, hold, start/end frame and score. One evaluation module scores all of them.
- **Versioned intermediate data:** pose and 3D distance files are exported once in their own
  environments (`external/pose2d`, `external/wham`), stored under a version name (official pose version
  `rtmw-x-384_2026-11-18`), shared through Google Drive and selected in `configs/paths.yaml`.
- **Repository layout:**
  - Reusable modules live in `src/holdon/`: `data`, `preprocessing`, `features`, `models`, `training`,
    `evaluation`, `visualization`.
  - `scripts/` only parses arguments.
  - `configs/` holds all paths and hyperparameters (YAML).
  - `tests/` holds CPU-only unit tests on synthetic data.
  - Each member owns one line and one long-lived branch; branches merge into `main` through pull requests.

## 3. Implementation

### 3.1 Data and protocol

- **Data:** The Way Up (Maschek & Schedl, CVPRW 2025): 22 videos (11 recordings × 2 routes), 720×1280 at
  25 fps. They contain 939 labelled contact events: 481 hand and 458 foot, including 31 wall pushes with
  no hold.
- **Splits:** leave-one-participant-out, 10 folds (recordings p2a and p2b are the same climber).
  Every tuned value is chosen on training participants only.
- **Metric:** a prediction matches a label only with the same limb and the same hold. Pairs are matched
  greedily by score under a temporal IoU (tIoU) threshold.
  - We report precision, recall and F1 at tIoU > 0, ≥ 0.3 and ≥ 0.5, as the macro average over the 22
    videos, plus the mean tIoU of matched pairs.

### 3.2 2D pose pipeline (2D line)

- **Climber selection:** YOLOX-m detects people in every frame. The climber is a detected person with at
  least 50% of the box inside the route region (the holds' bounding box plus 80 px), tracked by IoU.
- **Pose models:** RTMW-x (133 whole-body keypoints including fingertips and toes) and ViTPose++-H
  (17 keypoints, the paper's baseline) run on the climber's box.
- **Compute:** all 22 videos (38,692 frames) with both models took about 2 h 45 min on a GCP T4 GPU
  (RTMW 14 fps, ViTPose 5 fps).
- **Validation:** in labelled contact frames where the climber is detected, the RTMW wrist or ankle is
  within 40 px of the labelled hold 95.7% of the time.
  - Without the route-region filter this was 89.3%, because the tracker switched to the belayer (Figure 1).
  - ViTPose++-H on the same boxes: 97.1%. Frame counts match the videos in all 22 files, and the climber
    is detected in 92.2% of labelled contact frames.

![Figure 1](../results/figures/pose_selection_fix.jpg)
*Figure 1. Climber selection before (left, the tracker jumps to the belayer) and after the route-region filter (right).*

### 3.3 Track 1 — classical 2D baseline (2D line)

- **Rule:** a limb touches a hold if any of its keypoints is within `distance_px` of the hold box. In each
  frame the limb is assigned to the nearest such hold.
- **Event decoding:** runs separated by at most `merge_gap_frames` are merged, and contacts shorter than
  `min_duration_s` are dropped.
- **Keypoint processing:** low-confidence keypoints count as missing, gaps of up to 5 frames are
  interpolated, and tracks are median-smoothed.
- **Keypoint sets:** wrist/ankle (as in The Way Up) and fingertip/toe (5 fingertips per hand; big toe,
  small toe and heel per foot).
- **Tuning:** in each LOPO fold, the best of 480 parameter combinations is chosen on the training
  participants (objective: mean F1 over the three tIoU thresholds). Values chosen per setting:

| Pose, keypoints | distance_px | min_duration_s | merge_gap_frames | min_score |
| --- | --- | --- | --- | --- |
| RTMW fingertip/toe | 20 (9 of 10 folds) | 2.0 (10 of 10) | 40 (9 of 10) | 1.5 (8 of 10) |
| RTMW wrist/ankle | 45 (9 of 10) | 2.0 (9 of 10) | 25 (8 of 10) | 0.25 (9 of 10) |
| ViTPose wrist/ankle | 30 (10 of 10) | 2.0 (10 of 10) | 40 (10 of 10) | 0.05 (10 of 10) |

  RTMW scores range 0–10 and ViTPose scores 0–1, so `min_score` is set per model. The paper's untuned
  0.5 s rule is far too short here (see §5).
- **Demo videos** (`scripts/make_overlay.py`, held-out predictions only): `track1_p8_orange_0-39s.mp4`
  shows a whole climb, and `track1_p2a_green_17-27s.mp4` shows the false positive of Figure 3 in motion.
  Each frame shows the hold boxes, the labelled contact holds (orange), the Track 1 contact holds
  (magenta), the fingertip/toe keypoints (red), and a band with the labelled vs predicted hold per limb
  (green row = agree, red = disagree).

### 3.4 Evaluation module (learning line)

- **Shared scoring:** `holdon.evaluation` holds the shared scoring code: tIoU, greedy matching,
  per-video counts and macro P/R/F1.
  - Its `evaluate(pred_by_video, gt_by_video, thresholds, limbs)` function and `scripts/evaluate.py`
    score any track's `events.csv`.
  - The code was moved unchanged from the Track 1 tuning script. Track 1's outputs are reproduced
    byte for byte.
- **Breakdowns:**
  - hands vs feet;
  - per route;
  - occluded vs visible frame recall, using the dataset's `framesOccluded` annotation;
  - climber experience, as group averages only.
- **Privacy:** only one participant answered "less than 1 year", so that category is merged with
  "1–2 years" and no reported number belongs to a single person.

### 3.5 Track 2 v1 — learned per-frame classifier (learning line)

- **Candidates:** scoring every limb against every hold in every frame would give about 4.0 M rows. For
  each frame and limb we keep only holds within 80 px of the fingertip/toe or wrist/ankle keypoints that
  are also among the 3 nearest holds.
  - This gives 394,759 rows and keeps 91.8% of the labelled contact frames.
- **Features (36 per row):**
  - 2D overlap: fingertip/toe and wrist/ankle distance to the box, inside-box flags, hold rank, holds
    nearby, keypoint confidence, hand or foot;
  - stillness: keypoint speed in climber-box heights per frame;
  - temporal context: rolling mean and minimum of distances and speeds over ±2, ±6 and ±12 frames.
- **Classifier:** median imputation → standardization → Nystroem RBF kernel approximation (300
  components) → linear SVM with balanced class weights. It is trained on a stratified sample of 60,000 rows.
- **Decoding:** in each frame, a limb's highest-scoring candidate hold counts as a contact if its score
  is above a threshold. Track 1's event decoding then applies (minimum duration, gap merging).
- **Protocol:** outer LOPO (10 folds).
  - Inside each fold, the 9 training participants are split into 3 groups.
  - The out-of-fold SVM scores choose the threshold, minimum duration and gap length.
  - The SVM is then refit on all training participants and applied to the held-out participant.
- **Runtime:** a full run takes about 3 minutes on a laptop CPU.

### 3.6 Track 3 — 3D pipeline (3D line)

**[3D line — to be filled in by Jinghong Yang:** WHAM inference on the climbing videos; wall pose from
the bolt-hole homography; per-frame 3D limb-to-hold distances; what runs and on which videos.**]**

## 4. Tools and libraries

- **Code:** Python 3.14; OpenCV 5 (video I/O, drawing); NumPy; matplotlib; PyYAML.
- **2D pose:** rtmlib 0.0.16 with ONNX Runtime GPU 1.32 (YOLOX-m detector, RTMW-x); HuggingFace
  transformers 5.23 with PyTorch 2.15 (ViTPose++-H); ffmpeg (H.264 demo videos).
- **Track 2:** scikit-learn (Nystroem, LinearSVC).
- **3D:** WHAM + SMPL (Track 3).
- **Compute:** Google Cloud (n1-standard-4, NVIDIA T4), plus a Colab notebook as fallback.
- **Development:** pytest (82 CPU-only unit tests on synthetic data); GitHub with one branch per line and
  pull requests; Google Drive for data and versioned outputs.

## 5. Initial results

### 5.1 Results table

Macro average over 22 videos, held-out LOPO predictions only, precision / recall / **F1**, RTMW pose
unless noted.

| Setting | tIoU > 0 | tIoU ≥ 0.3 | tIoU ≥ 0.5 |
| --- | --- | --- | --- |
| **Track 2 v1 (SVM), all limbs** | 0.947 / 0.887 / **0.915** | 0.928 / 0.869 / **0.896** | 0.883 / 0.828 / **0.853** |
| — hands | 0.923 / 0.883 / **0.899** | 0.891 / 0.854 / **0.869** | 0.829 / 0.796 / **0.809** |
| — feet | 0.973 / 0.895 / **0.931** | 0.967 / 0.890 / **0.925** | 0.945 / 0.870 / **0.904** |
| Track 1 fingertip/toe, all limbs | 0.890 / 0.848 / **0.864** | 0.869 / 0.827 / **0.844** | 0.814 / 0.773 / **0.790** |
| — hands | 0.834 / 0.832 / **0.828** | 0.799 / 0.796 / **0.792** | 0.726 / 0.721 / **0.719** |
| — feet | 0.959 / 0.868 / **0.908** | 0.956 / 0.865 / **0.905** | 0.923 / 0.835 / **0.874** |
| Track 1 wrist/ankle, all limbs | 0.757 / 0.755 / **0.752** | 0.737 / 0.734 / **0.731** | 0.654 / 0.649 / **0.648** |
| — hands | 0.856 / 0.848 / **0.847** | 0.835 / 0.826 / **0.825** | 0.762 / 0.745 / **0.749** |
| — feet | 0.656 / 0.662 / **0.652** | 0.638 / 0.644 / **0.634** | 0.550 / 0.554 / **0.546** |
| Track 1 ViTPose++-H wrist/ankle (paper setting) | 0.799 / 0.796 / **0.795** | 0.774 / 0.771 / **0.770** | 0.711 / 0.706 / **0.706** |
| — hands | 0.881 / 0.870 / **0.872** | 0.864 / 0.853 / **0.856** | 0.809 / 0.797 / **0.800** |
| — feet | 0.717 / 0.723 / **0.715** | 0.685 / 0.690 / **0.683** | 0.617 / 0.617 / **0.612** |
| Untuned rule (0.5 s, inside box), fingertip/toe | 0.548 / 0.761 / **0.631** | 0.370 / 0.512 / **0.426** | 0.219 / 0.306 / **0.254** |
| Track 3 | **[3D line — to be filled in]** | | |

For context only: The Way Up reports about 86.6% (tIoU > 0) and 69% (tIoU > 0.5) *accuracy* for its
ViTPose baseline. Their accuracy is defined differently from our F1, so the numbers are not directly
comparable.

### 5.2 Findings

1. **Toe keypoints fix the feet (Track 1).** Foot F1 at tIoU > 0 rises from 0.652 (ankle) to 0.908 (toes),
   because the ankle lies outside the foothold box when the climber stands on the toes.
2. **The duration rule matters most for Track 1.** Every fold chose a 2.0 s minimum duration for Track 1. The
   untuned 0.5 s rule reaches only F1 0.631 / 0.426 / 0.254.
   - The cost of the longer minimum: 137 of 939 contacts (14.6%, 66 of them hands) last under 2 s and
     cannot be found.
3. **Hands are the hard part for Track 1, and the pose model matters.**
   - The best Track 1 hand F1 is 0.872 (ViTPose wrist), below the best feet (0.908, RTMW toes).
   - For hands the wrist works better than the fingertips (RTMW 0.847 vs 0.828).
   - With the same boxes and wrist/ankle keypoints, ViTPose++-H beats RTMW-x (F1 0.795 vs 0.752 at
     tIoU > 0) at about a third of the speed. RTMW stays the main pose input because only it has
     fingertips and toes.
4. **Track 2 reduces false hand contacts, the research question.**
   - Hand precision at tIoU > 0 rises from 0.834 to 0.923.
   - Pooled over the 22 videos, false hand contacts drop from 91 to 38, while correct hand contacts
     rise from 401 to 422 of 481.
   - All-limb F1 is 0.05–0.06 higher than the best Track 1 setting at every threshold.
   - Track 2 chose a 1.0 s minimum duration, so only 40 contacts (4.3%) are too short to find.
5. **Occlusion.**
   - Track 2 recovers more occluded hand frames than Track 1 (frame recall 0.872 vs 0.779).
   - Occluded foot frames are worse (0.734 vs 0.845), but there are only 990 of them.
6. **Accurate intervals.** The mean tIoU of matched events is 0.882 for Track 2 (Track 1: 0.836), so
   detected contacts also have accurate start and end frames.
7. **No clear effect of climbing experience.** Track 2 F1 at tIoU > 0 is 0.912 for climbers with up to
   2 years (6 participants), 0.924 for 5–10 years (2) and 0.914 for over 10 years (2). The groups are small.
8. **Track 3.** **[3D line — to be filled in]**

![Figure 2](../results/figures/track1_overlay_p8_orange.jpg)
*Figure 2. Track 1 on p8_orange at 18.8 s. Orange: labelled contacts; magenta: Track 1 contacts; red: fingertip and toe keypoints; green / blue: hand / foot hold boxes. All four limbs are on the correct holds.*

![Figure 3](../results/figures/track1_false_positive.jpg)
*Figure 3. Typical false positive: p2a_green at 21.6 s, the right hand's fingertips lie inside hold 11's box for 4.2 s without a labelled contact. Track 2 still predicts this contact (21.7–25.7 s), because the hand is close and still; telling "on" from "in front of" the hold needs depth, which is what Track 3 adds.*

![Figure 4](../results/figures/track2_timeline_p8_orange.png)
*Figure 4. Labelled (orange) vs Track 2 (magenta) contacts per limb over the whole climb of p8_orange (held-out predictions); numbers are hold ids. Most contacts match in hold and timing; the remaining errors are short extra contacts where the hand moves between holds (left hand 16 → 18, right hand 27 → 28).*

## 6. Challenges and planned improvements

| Line | Challenge | Status / plan |
| --- | --- | --- |
| 2D | Tracker switched to the belayer when the climber was missed for a frame | Route-region filter; 95.7% of contact frames near the labelled hold |
| 2D | Climber not detected high on the wall (20–30% of contact frames missing in three videos) | Phase 3: larger detector input, tracking through missed frames |
| 2D | Occluded fingertip keypoints drift (hands are more than 50% occluded in about half of their contact frames) | Track 1 occluded hand frame recall 0.779 vs 0.869 visible; Track 2 and 3D cues target it |
| 2D | GPU quota 0, then T4/L4 out of stock in every US zone | Quota raised, automatic retries got a T4; Colab notebook as fallback |
| 2D / learning | Tuning grids chose values at their edge | Grids widened until the choice was inside (choices on training folds only) |
| Learning | 8.2% of labelled contact frames are unreachable for Track 2 (6.3% no pose, 2.0% hold not a candidate) | Better climber detection and 3D distances should recover part of it |
| Learning | Per-frame SVM decisions flicker inside contacts; decoding merges gaps of up to 3.2 s | Phase 3: temporal model (GRU / 1D-CNN) over feature windows |
| Learning | A still hand in front of a hold is still a false positive (Figure 3) | Phase 3: 3D distance features from Track 3 |
| 3D | **[3D line — to be filled in]** | |

**Phase 3 plan:**

- **2D line:** hybrid keypoints (ViTPose wrist for hands, RTMW toes for feet); fewer missed climber
  frames; faster pose export (fp16, batching); Kilter board clips with a labeling script and LED-colour
  hold detection for a cross-dataset test.
- **Learning line:** add the 3D distance features; train a temporal model; run an ablation (remove 2D
  overlap, stillness, context and 3D cues one at a time); build a failure gallery.
- **3D line:** **[to be filled in]**.
- **All lines:** a final three-track comparison on the same splits.

## 7. Contributions

| Member | Line | Phase 2 work |
| --- | --- | --- |
| Zongyang Li | 2D line | Data and Drive distribution; 2D pose export (RTMW, ViTPose++) with climber selection, GCP runs and validation; Track 1 baseline and fingertip/toe variant with LOPO tuning and the first scoring code (now in the evaluation module); ViTPose comparison; interfaces and documentation; figures, Track 1 demo videos and slide material |
| Yen-Wei Lin | Learning line | Evaluation module (shared scoring, `evaluate()`, hands/feet, route, occlusion and experience breakdowns, `scripts/evaluate.py`); Track 2 v1 (candidate limb–hold pairs, 2D overlap + stillness + context features, Nystroem SVM, LOPO with inner-fold tuning); Track 2 figures and results; unit tests for all new modules; Phase 2 progress report draft; presentation slides |
| Jinghong Yang | 3D line | **[to be filled in:** WHAM inference; wall pose from the homography; per-frame 3D distances**]** |
