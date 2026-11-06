# Project Context — Team 06 "Hold On"

CS 5330 Pattern Recognition and Computer Vision (Northeastern, Prof. Akram Bayat), Fall 2026.
Team: Zongyang Li (lead), Yen-Wei Lin, Jinghong Yang.

This file is the single source of project context for anyone (human or AI) working in this repository.

---

## 1. Goal

Given a monocular climbing video, detect **which hand or foot touches which hold, and when**.

- **Input:** a climbing video plus hold bounding boxes.
- **Output:** contact events `(video_id, limb, hold_id, start_frame, end_frame, score)`, a contact timeline, and an overlay video.
- **Problem types:** Detection (holds, hand/foot keypoints) + Classification (contact or not, per frame, limb and hold) + pose estimation (2D and monocular 3D).

**Core research question:** existing 2D methods already reach high recall; their main error is false positives (a hand passing in front of a hold counts as contact, worse on densely set routes). Using monocular video only, how far can temporal and 3D cues reduce false contacts, especially for hands?

### Three tracks (compared under identical splits and evaluation)

| Track | Method | Training |
| --- | --- | --- |
| 1 Classical 2D baseline | 2D keypoint inside a hold box for ≥ 0.5 s = contact (reproduces The Way Up). Plus a fingertip/toe keypoint variant. | None; thresholds tuned on training split only |
| 2 Learned temporal classifier | Per-frame features (2D overlap, limb stillness, 3D limb-to-wall distance) → temporal model (SVM reference; GRU or 1D-CNN) | Trained |
| 3 3D pipeline | Pretrained WHAM gives the 3D body; wall pose from decomposing the per-video homography; 3D hand/foot-to-hold distance decides contact | None; pretrained models |

Track 2 consumes Track 3's 3D distances, so Track 3 inference must run early.

### Final outputs

- Contact-detection pipeline producing timelines and overlay videos
- Three-track comparison: Precision, Recall, F1 at tIoU thresholds 0 / 0.3 / 0.5
- Ablation: remove 2D, stillness, 3D cues one at a time (within Track 2)
- Grouped analysis: hands vs feet, occluded vs visible, by route, by climber experience
- Failure-case gallery
- Demo on phone-recorded Kilter board videos (generalization test)
- Stretch: use confirmed contacts to refine 3D pose; CIMI4D evaluation if the dataset becomes reachable

---

## 2. Data

### 2.1 The Way Up (main dataset)

- Paper: Maschek & Schedl, *The Way Up: A Dataset for Hold Usage Detection in Sport Climbing*, CVPR Workshops 2025 (arXiv 2505.12854).
- Download: Zenodo, DOI 10.5281/zenodo.15196866 — single `dataset.zip`, 20.9 GB. License CC BY 4.0 (redistribution allowed with attribution).
- Content: 22 videos (11 participant recordings of 10 climbers × 2 routes: "orange" 4c and "green" 5a), 8.5 m high 0° vertical planar indoor wall, 940 hold usages (481 hands, 459 feet), ~1547 s total. Hands are >50% occluded in roughly half of their contact frames; feet ~2%.
- Labels were made by a climbing coach on the **processed** videos.

**Known file layout** (from the Zenodo file browser; verify against the real files):

```
dataset.zip
├── p1/ … p11/                      # one folder per participant recording (verify exact names/count)
│   ├── green.mp4                   # processed: 720×1280 (portrait), 25 fps, stabilized — USE THIS
│   ├── green_large.mp4             # 4K / 50 fps original — NOT needed, skip
│   ├── green_holdUsage.csv         # contact labels: limb, hold, start/end frame, occluded ranges
│   ├── green_routeAnnotation.json  # hold bounding boxes (may include homography; verify)
│   ├── green_bolt_holes.json       # reference points for the homography
│   └── orange.mp4, orange_large.mp4, orange_holdUsage.csv, orange_routeAnnotation.json, orange_bolt_holes.json
├── participant_form_data.csv       # participant demographics and experience; grouped analysis only
├── topo_green.png
└── topo_orange.png
```

**Column names and JSON schemas are not yet confirmed.** The first data task is to inspect the real files and document their exact schema in `docs/` before writing loaders. Do not guess field names.

**Distribution:** everyone gets the data from the team Google Drive as `wayup.zip` (546.5 MB, 92 files: the original files minus `*_large.mp4`), installed with `scripts/install_data.py`. See the README.

**First sanity check:** draw the hold boxes from `*_routeAnnotation.json` on the first and last frame of each processed video and confirm they line up.

### 2.2 Self-recorded Kilter board videos (generalization + demo)

- ~10–15 clips at Movement gym (Kilter board: adjustable angle, overhanging, holds lit by LEDs). Recorded by Zongyang.
- Phone on a fixed tripod, landscape 1080p at 60 fps, focus/exposure locked, all four board corners in frame, board angle noted per session. Consent from everyone filmed; no bystanders.
- Phones often record variable frame rate: convert to constant frame rate with ffmpeg before labeling.
- Labels: same format as The Way Up, so one evaluation script covers both. Hold boxes are drawn once per video (static camera); a small OpenCV labeling script (`cv2.selectROIs` + keypresses for start/end frames) writes the CSV.

### 2.3 Splits

Leave-one-participant-out cross-validation (LOPO). Never split by random frames. All three tracks use identical splits. Hyperparameters are tuned only on training folds.

### 2.4 CIMI4D

Optional (stretch). Its website is currently unreachable. Do not build anything that depends on it.

---

## 3. Evaluation protocol (shared by all tracks)

- **Event-level matching:** a predicted event may only match a ground-truth event with the **same limb and same hold**. Among eligible pairs, compute temporal IoU `tIoU = |Tp ∩ Tg| / |Tp ∪ Tg|` on frame intervals. Greedy matching ordered by prediction score; each event used at most once.
- TP = matched prediction, FP = unmatched prediction, FN = unmatched ground truth. No true negatives (most holds are never used).
- Report Precision, Recall, F1 at tIoU > 0, ≥ 0.3, ≥ 0.5, plus mean tIoU over matched events.
- Macro-average over videos.
- Breakdowns: hands vs feet, by route, occluded vs visible (occlusion-stratified frame recall using the dataset's occlusion ranges), by climber experience.
- Reference numbers: The Way Up's best baseline (ViTPose) reports ~86.6% accuracy at tIoU > 0 and ~69% at tIoU > 0.5 (their accuracy definition differs from F1; compare carefully). The 2026 "Training-Free" preprint (arXiv 2609.30026) has internally inconsistent result tables — borrow its protocol ideas, do not treat its numbers as a target.

---

## 4. Modules and techniques

| Module | Work | Main techniques |
| --- | --- | --- |
| M1 Data | Load The Way Up; schema docs; alignment check; Kilter recording conversion and labeling script | OpenCV video I/O, ffmpeg, `cv2.selectROIs` |
| M2 Track 1 | Whole-body 2D pose; hold-box overlap + 0.5 s persistence rule; fingertip/toe variant | MMPose (RTMPose-WholeBody) or ViTPose; trajectory smoothing (Gaussian/median) |
| M3 Evaluation | Event matching, P/R/F1, grouped stats, timelines and overlay videos | NumPy, pandas, matplotlib |
| M4 Track 3 | WHAM inference (fixed camera → `--estimate_local_only`); wall pose via `cv2.decomposeHomographyMat`; per-frame 3D limb-to-wall/hold distance | WHAM, SMPL, homography + RANSAC |
| M5 Track 2 | Features (2D overlap, stillness via optical flow or keypoint velocity, 3D distance); classifier training; ablation | scikit-learn SVM reference; PyTorch GRU or 1D-CNN |
| M6 Kilter + demo | Classical hold detection (LED colour HSV threshold + morphology + connected components) vs fine-tuned YOLO; phone-video demo | OpenCV, Ultralytics YOLO |

**Key technical risk (Track 3):** The Way Up camera intrinsics are unknown. Use an estimated focal length and validate the recovered wall plane against known hold spacing; fallback is a 2.5D distance in wall coordinates.

**WHAM environment:** heavy dependencies and needs a GPU (Colab is fine). Keep WHAM in its own environment (or Colab notebook) and have it **export per-video outputs to files** (e.g. `.npz` with SMPL joints/vertices per frame). The main package only reads those files and never imports WHAM directly.

---

## 5. Roles (placeholders until confirmed)

Each person owns one line end to end:

- **A — 3D line:** Track 3 (M4), system integration, demo video
- **B — 2D line:** data (M1), Track 1 (M2), fingertip variant, Kilter labels, dataset docs
- **C — Learning line:** evaluation (M3), Track 2 (M5), ablation, grouped analysis, failure gallery
- **Zongyang:** Kilter video recording (in addition to her line)

Every member commits from their own GitHub account; contributions must be traceable through commits.

---

## 6. Course requirements that shape the repository

### Phase 2 — Implementation and Initial Testing (due Nov 23, 2026)

- Working prototype of the core functionality; data loading, preprocessing, feature extraction and model components as applicable; must handle representative input data.
- Any language compatible with OpenCV.
- Test on sample data; report preliminary results (metrics or visual outputs), challenges and limitations.
- Deliverables: prototype, source code and supporting files, progress report + presentation (overview, architecture, tools/libraries, initial results with screenshots, challenges, per-member contributions).

### Phase 3 — Final Implementation, Evaluation, Submission (due Dec 8, 2026)

- **Code organized into reusable modules:** data loading, data preprocessing, feature extraction, model definition, model training, evaluation and testing.
- Testing: **unit tests** and validation on different data subsets; relevant metrics; strengths, limitations, comparison with original objectives.
- Public (or instructor-accessible) GitHub repo with a **README** containing: title and purpose; problem statement and objectives; environment setup and run instructions; dependencies and installation; description of each module and script; instructions to reproduce results; special considerations / known issues.
- **Dataset:** large data goes to cloud storage (e.g. Google Drive); link, description of contents and structure, and all preprocessing/transformations documented in the README.
- **Version control:** regular commits with meaningful messages; branches for features/experiments; each member's contributions identifiable through commits and docs.

---

## 7. Phase plan

| Phase | A: 3D line | B: 2D line | C: Learning line | Due |
| --- | --- | --- | --- | --- |
| 0 Setup | WHAM running on one clip in Colab | Download data subset; schema docs; alignment check; data loader | Repo skeleton; evaluation protocol spec | Before Nov 9 |
| 1 Proposal | Slides/video | Slides/video | Slides/video | Nov 9 |
| 2 First results | WHAM inference; wall pose; per-frame 3D distances | Reproduce Track 1 vs the paper's numbers; Kilter labeling script | Evaluation module; Track 2 v1 (2D overlap + stillness) | Nov 23 |
| 3 Refinement | Finalize 3D distances; Track 3 on Kilter; timeline/overlay visualizations | Fingertip variant; Kilter labels; cross-dataset test | Track 2 with 3D features; ablation; grouped analysis | First half of Phase 3 |
| 4 Delivery | Integration; demo video | Dataset description + drive link; data unit tests | Final tables; failure gallery; evaluation unit tests | Dec 8 |

---

## 8. References

1. Maschek & Schedl, The Way Up, CVPR Workshops 2025 — arXiv 2505.12854
2. Shin et al., WHAM, CVPR 2024 — arXiv 2312.07531, github.com/yohanshin/WHAM
3. Goel et al., HMR 2.0 / 4D-Humans, ICCV 2023 — github.com/shubham-goel/4D-Humans
4. Tripathi et al., DECO, ICCV 2023 — github.com/sha2nkt/deco (optional Track 2 feature)
5. Yan et al., CIMI4D, CVPR 2023 — arXiv 2303.17948; Yan et al., ClimbingCap, CVPR 2025 — arXiv 2503.21268
6. Bakar et al., Training-Free Hold-Usage Detection with Foundation Pose Models, arXiv 2609.30026 (protocol reference only)

Tools: MMPose, ViTPose, WHAM + SMPL/SMPLify (accounts registered by Zoe), Ultralytics YOLO, OpenCV, PyTorch, NumPy, pandas.
