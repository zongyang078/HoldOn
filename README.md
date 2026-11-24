# Hold On — Hand/Foot–Hold Contact Detection from Monocular Climbing Video

CS 5330 Pattern Recognition and Computer Vision, Fall 2026 — Team 06
(Zongyang Li, Yen-Wei Lin, Jinghong Yang).

Given a climbing video and hold bounding boxes, detect **which hand or foot touches which hold, and when**.
2D methods already find most contacts but also report false ones (a hand passing in front of a hold).
We compare three tracks under the same splits and metrics:

1. **Track 1 — classical 2D baseline:** a keypoint near a hold box for long enough is a contact (thresholds tuned with LOPO).
2. **Track 2 — learned classifier:** 2D overlap, stillness and temporal-context features → SVM (v1); 3D distance
   features and a GRU / 1D-CNN follow in Phase 3.
3. **Track 3 — 3D pipeline:** WHAM body + wall pose from the bolt-hole homography → 3D limb-to-hold distance.

**Phase 2 status** (macro F1 over 22 videos, held-out LOPO predictions, tIoU > 0 / ≥ 0.5): Track 1 fingertip/toe
0.864 / 0.790; Track 2 v1 0.915 / 0.853, with false hand contacts down from 91 to 38; Track 3 (WHAM runner, wall
pose, distance files) in progress on `feature/3d-line`. Details: [docs/phase2_results.md](docs/phase2_results.md),
report: [docs/phase2_report.md](docs/phase2_report.md).

Background and plan: [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).
**How the lines connect (formats, APIs, versions): [docs/interfaces.md](docs/interfaces.md).**

## Quick start

```bash
# 1. Environment (Python 3.14, exact versions)
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip3 install -r requirements.txt
python3 -m pytest                                      # passes in a few seconds, no data needed

# 2. Dataset: download wayup.zip (546.5 MB) from the team Drive, HoldOn/data/raw/ (link below)
python3 scripts/install_data.py ~/Downloads/wayup.zip   # checks SHA-256, extracts to data/raw/wayup/
python3 scripts/check_alignment.py                     # optional: hold boxes on first/last frames

# 3. 2D pose: download the folders named in configs/paths.yaml → versions from
#    HoldOn/data/interim/pose2d/ on the Drive into data/interim/pose2d/ (same folder names)

# 4. Track 1 (needs steps 2 and 3)
python3 scripts/tune_track1.py                         # LOPO-tuned metrics → results/track1/<run>/metrics.md

# 5. Track 2 v1 (needs steps 2 and 3; CPU, about 3 min)
python3 scripts/run_track2.py                          # → results/track2/<run>/metrics.md
```

Drive: [data/](https://drive.google.com/drive/folders/14LDiwWxDYSHDEanNLgf-o8uWpW3GYr23) ·
[wayup.zip](https://drive.google.com/file/d/1GYdSKut77Rjo-WNFYzZ4okUVW3kSMDgr/view) ·
[deliverables/](https://drive.google.com/drive/folders/1XWcjto1pSWkWLWmeqXtztriikkjzU9bY) ·
[team root](https://drive.google.com/drive/folders/141TIhRlO4kW7jZ-73BvQd77yGjOUokYU) (team only).
Download files to your disk; do not point `data/` at a synced Drive folder (slow, sync conflicts).

## Where each member starts

| Member, branch | Builds | Reads | Delivers (format in [interfaces.md](docs/interfaces.md)) |
| --- | --- | --- | --- |
| Zongyang Li, `feature/2d-line` | data, 2D pose export, Track 1, Kilter clips/labels/hold detection | — | pose files (§3), Track 1 events (§4) |
| Yen-Wei Lin, `feature/learning-line` | evaluation, Track 2, ablation, grouped analysis, failure gallery | pose files, distance files, events | `holdon.evaluation` + `scripts/evaluate.py` (§5), Track 2 events (§4) |
| Jinghong Yang, `feature/3d-line` | Track 3 (WHAM, wall pose, 3D distances), integration, demo | videos, `rec.wall` homography | distance files (§6), Track 3 events (§4) |

- **Learning line:** scoring lives in `holdon.evaluation` (`scripts/evaluate.py` scores any `events.csv`,
  interfaces §5); Track 2 v1 (2D overlap + stillness → SVM) runs with `distance3d: false` in
  `configs/track2.yaml` until distance files exist.
- **3D line:** WHAM runs on the 2D line's ViTPose climber keypoints (`external/wham/run_wham.py`, Colab);
  wall pose and distance files (interfaces §6) are on `feature/3d-line`, to be merged; then set
  `versions.distance3d` in `configs/paths.yaml` so Track 2 can add 3D features.

## Repository layout

| Path | Contents | Status |
| --- | --- | --- |
| `src/holdon/data/` | The Way Up loader, video I/O, pose / distance file readers, LOPO splits | done |
| `src/holdon/preprocessing/` | keypoint smoothing (done); homography / wall pose (on `feature/3d-line`) | partly |
| `src/holdon/features/` | 2D keypoint-to-hold distances, candidate holds, keypoint stillness, Track 2 feature tables (done); 3D distances (on `feature/3d-line`) | partly |
| `src/holdon/models/` | contact rules + event decoding (done, shared by all tracks), Track 1 tuning, Track 2 SVM (done); GRU / 1D-CNN, hold detection | partly |
| `src/holdon/evaluation/` | event matching, P / R / F1, breakdowns (hands/feet, route, occlusion, experience), report tables (done); ablation | partly |
| `src/holdon/training/` | Track 2 LOPO training with inner-fold decoding tuning (done); windows for GRU | partly |
| `src/holdon/visualization/` | timelines, frame overlays, per-limb status band for the demo videos | done |
| `src/holdon/events.py` | shared contact-event format | done |
| `scripts/` | thin command-line entry points (logic stays in `src/holdon/`): data install, pose checks, Track 1 tuning and figures, demo videos, evaluation, Track 2 | — |
| `external/pose2d/` | 2D pose export, separate environment ([README](external/pose2d/README.md)) | done |
| `external/wham/` | WHAM, separate environment ([README](external/wham/README.md)); runner on `feature/3d-line` | in progress |
| `configs/` | paths, official data versions, per-track hyperparameters (YAML) | — |
| `docs/` | [interfaces](docs/interfaces.md), [dataset schema](docs/dataset_schema.md), [pose format](docs/pose2d_format.md), [events](docs/event_format.md), [evaluation](docs/evaluation_protocol.md) | — |

Empty modules (a one-line docstring) are planned work of the owning line; fill them in, rename or delete
them as needed.

## Data

- **The Way Up** (Maschek & Schedl, CVPR Workshops 2025, CC BY 4.0): 22 videos (11 recordings × 2 routes),
  720×1280, 25 fps, 939 labeled contacts. `wayup.zip` = the original `dataset.zip` (Zenodo,
  DOI 10.5281/zenodo.15196866) minus the 22 `*_large.mp4` 4K files; nothing else changed. Formats and quirks:
  [docs/dataset_schema.md](docs/dataset_schema.md).
- **Derived data** (pose, WHAM, distances, features, checkpoints) lives on the Drive under the same paths as
  in `data/`, one folder `<name>_<YYYY-MM-DD>/` per version with a `MANIFEST.md`
  ([template](docs/drive_manifest_template.md)). Never overwrite a version; the versions every track uses are
  listed in `configs/paths.yaml` → `versions`.
- **`participant_form_data.csv`** (inside `wayup.zip`) holds demographics: use it only for grouped analysis,
  never report individual values or link them to people in videos or figures.
- Maintainers: `python3 scripts/package_wayup.py` rebuilds `wayup.zip` deterministically from `dataset.zip`;
  update `shared.wayup_zip_sha256` in `configs/paths.yaml` when it changes.

## Reproducing results

| Result | Command | Output |
| --- | --- | --- |
| 2D pose (GPU) | [external/pose2d/README.md](external/pose2d/README.md) | `data/interim/pose2d/<version>/` |
| Track 1, RTMW wrist/ankle | `python3 scripts/tune_track1.py` | `results/track1/rtmw-x-384_2026-11-18_wrist_ankle/metrics.md` |
| Track 1, fingertip/toe | `python3 scripts/tune_track1.py --keypoints fingertip_toe` | `results/track1/…_fingertip_toe/` |
| Track 1, ViTPose (paper) | `python3 scripts/tune_track1.py --pose-version vitpose-plus-huge_2026-11-18` | `results/track1/vitpose-plus-huge_2026-11-18_wrist_ankle/` |
| Track 1 figures | `python3 scripts/make_track1_figures.py` (after the fingertip/toe run) | `results/figures/track1_*` |
| Track 1 demo video | `python3 scripts/make_overlay.py --video p8_orange --scale 0.75` (after the fingertip/toe run) | `results/videos/track1_*.mp4` |
| Score any run (all / hands / feet, route, occlusion, experience) | `python3 scripts/evaluate.py results/<track>/<run>/events_lopo.csv` | `metrics.md` next to the events file (`--out` to rename) |
| Track 2 v1, SVM on 2D overlap + stillness (CPU, ~3 min) | `python3 scripts/run_track2.py` | `results/track2/rtmw-x-384_2026-11-18_v1/metrics.md` |
| Track 3 | _to be added by the 3D line (`feature/3d-line`)_ | |

## Known issues

- Camera intrinsics of The Way Up are unknown; Track 3 fits one focal length to the bolt holes of all videos.
- 20 of the 22 `*_bolt_holes.json` files list the two top bolt holes in swapped order, so their stored
  `homography_mat` is wrong away from the four points. Track 1 and Track 2 do not use it; the 3D line repairs
  the order before computing the wall pose (`feature/3d-line`).
- `tests/data/test_archive.py::test_check_layout` fails on Windows only (path separator); all other tests pass there.
- SMPL files (needed by WHAM) are licensed per person: whoever runs WHAM registers at smpl.is.tue.mpg.de.
  No other model needs an account; pose weights download automatically.
- 2D pose issues and their fixes (belayer tracking, GPU setup): [external/pose2d/README.md](external/pose2d/README.md).

## Team conventions

- `python3` / `pip3` in all commands; code, comments, docs and commit messages in English.
- Python 3.10+, type hints on public functions, a docstring on every module and public function.
- Logic in `src/holdon/`; `scripts/` only parse arguments and call the package. Paths and hyperparameters in `configs/`.
- Tests: `pytest`, CPU, seconds, tiny synthetic data; never the real dataset or a GPU.
- Never commit data, videos, weights or large outputs (`data/`, `checkpoints/`, `results/` are git-ignored).
- Splits are leave-one-participant-out; never split by random frames. Tune on training folds only.
- WHAM and the 2D pose models run in their own environments and write files; the package only reads them.
- Prefer OpenCV-compatible implementations (course requirement). Ask the team before adding a heavy dependency.

### Git

- Each member commits from their own GitHub account on their **line branch** (table above).
- `main` is maintained by the team lead; nobody pushes to it directly. Merge your branch into `main` through
  a pull request at every milestone (at least before each phase deadline, and whenever someone needs your
  code). Run `python3 -m pytest` first.
- After a merge, update your branch: `git switch <your-branch> && git pull && git merge origin/main`.
- Extra short branches only for risky experiments (`exp/<topic>`) or urgent fixes to `main` (`fix/<topic>`).
- Changing anything marked FIXED in [docs/interfaces.md](docs/interfaces.md): say so in the PR description.

## License and attribution

- **Code:** MIT License, see [LICENSE](LICENSE).
- **The Way Up dataset:** © Maschek & Schedl, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Our
  Drive copy is the unmodified processed subset; all derived files are our own.
- **Third-party models:** WHAM, SMPL, RTMW (OpenMMLab), ViTPose and Ultralytics YOLO keep their own licenses;
  we do not redistribute their weights.

```bibtex
@inproceedings{maschek2025wayup,
  title     = {The Way Up: A Dataset for Hold Usage Detection in Sport Climbing},
  author    = {Maschek, Anna and Schedl, David C.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2025},
  note      = {arXiv:2505.12854, dataset DOI 10.5281/zenodo.15196866}
}
```

## Contributions

| Member | Line | Owns |
| --- | --- | --- |
| Zongyang Li | 2D line | Data (The Way Up loading, Drive distribution), Track 1 baseline and fingertip variant, Kilter recording, labels and hold detection |
| Yen-Wei Lin | Learning line | Evaluation module, Track 2 classifier, ablation, grouped analysis, failure gallery |
| Jinghong Yang | 3D line | Track 3 (WHAM, wall homography, 3D distances), system integration, demo video |

### Phase 2 (implementation and initial testing)

- **Zongyang Li:** data and Drive distribution; 2D pose export (RTMW, ViTPose++) with climber selection, GCP runs and validation; Track 1 baseline and fingertip/toe variant with LOPO tuning; first scoring code (now in `holdon.evaluation`); ViTPose comparison; interfaces and documentation; figures, Track 1 demo videos and slide material.
- **Yen-Wei Lin:** evaluation module (scoring moved from Track 1 with identical numbers, `evaluate()`, hands/feet, route, occlusion and experience breakdowns, `scripts/evaluate.py`); Track 2 v1 (candidate limb–hold pairs, 2D overlap + stillness + temporal-context features, Nystroem SVM, LOPO with inner-fold tuning, `scripts/run_track2.py`); Track 2 results and figures; Phase 2 progress report draft; presentation slides.
- **Jinghong Yang:** WHAM runner on the 2D line's climber keypoints (coverage on `p3_orange` 26% → 96% vs WHAM's own tracker); wall pose with one shared focal length and the bolt-hole order fix; 3D limb-to-hold distance code (`feature/3d-line`).

Results and report material: [docs/phase2_results.md](docs/phase2_results.md).

### Phase 1 (proposal)

- **Zongyang Li:** GitHub repository and environment; dataset setup (Drive sharing, loader, schema docs); dataset and sample slides.
- **Yen-Wei Lin:** evaluation protocol (metrics, leave-one-participant-out splits); related-work review; project plan.
- **Jinghong Yang:** problem definition; three-track method design; 3D feasibility study (WHAM, wall homography).
