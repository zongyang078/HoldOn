# Hold On — Hand/Foot–Hold Contact Detection from Monocular Climbing Video

CS 5330 Pattern Recognition and Computer Vision, Fall 2026 — Team 06
(Zongyang Li, Yen-Wei Lin, Jinghong Yang).

## Purpose and problem statement

Given a monocular climbing video and hold bounding boxes, detect which hand or foot touches
which hold, and when. Output: contact events `(video_id, limb, hold_id, start_frame, end_frame, score)`,
a contact timeline and an overlay video.

Existing 2D methods reach high recall but produce false contacts (a hand passing in front of a hold).
We study how far temporal and 3D cues reduce those false positives, comparing three tracks:

1. **Classical 2D baseline** — keypoint inside a hold box for >= 0.5 s.
2. **Learned temporal classifier** — 2D overlap, stillness and 3D distance features into SVM / GRU / 1D-CNN.
3. **3D pipeline** — WHAM body + wall pose from homography; 3D limb-to-hold distance.

Full context: [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).

## Repository structure

| Path | Contents |
| --- | --- |
| `src/holdon/data/` | Data loading (videos, labels, external model outputs, LOPO splits) |
| `src/holdon/preprocessing/` | Smoothing, homography / wall pose, label conversion |
| `src/holdon/features/` | 2D overlap, stillness, 3D distance, feature tables |
| `src/holdon/models/` | Rule-based detectors, SVM, GRU / 1D-CNN, event decoding, hold detectors |
| `src/holdon/training/` | Windowed datasets, LOPO training, threshold tuning |
| `src/holdon/evaluation/` | Event matching, metrics, grouped analysis, ablation |
| `src/holdon/visualization/` | Timelines and overlay videos |
| `src/holdon/events.py` | Shared contact-event format |
| `scripts/` | Thin command-line entry points |
| `external/` | WHAM and 2D pose, run in separate environments and exported to files |
| `configs/` | Paths and hyperparameters (YAML) |
| `tests/` | Unit tests (pytest, CPU, synthetic fixtures) |
| `docs/` | Project context, dataset schema, event format, evaluation protocol |

## Setup

Use **Python 3.14** (see `.python-version`; pyenv and uv pick it up automatically).
`requirements.txt` pins exact versions, so every member gets the same environment.
The `.venv/` folder itself is not committed; each member creates it locally:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip3 install -r requirements.txt
python3 -m pytest                  # should pass in a few seconds
```

When you add or upgrade a dependency, add it to `pyproject.toml`, regenerate the pinned block in
`requirements.txt` with `pip3 freeze`, and commit both in the same pull request.

WHAM and 2D pose use their own environments; see `external/wham/README.md` and `external/pose2d/README.md`.

## Dataset

All data for this project is distributed through the **team Google Drive**:
[`HoldOn/data/`](https://drive.google.com/drive/folders/14LDiwWxDYSHDEanNLgf-o8uWpW3GYr23). Nothing is stored in git.

Main dataset: *The Way Up* (Maschek & Schedl, CVPR Workshops 2025), CC BY 4.0 — see
[License and attribution](#license-and-attribution). We use only the processed videos
(`green.mp4`, `orange.mp4`, 720×1280, 25 fps).

### Getting the data

1. Download [`wayup.zip`](https://drive.google.com/file/d/1GYdSKut77Rjo-WNFYzZ4okUVW3kSMDgr/view) (546.5 MB) from `HoldOn/data/raw/` on the team Drive.
2. Install it (verifies the SHA-256 checksum, extracts, checks all 92 files are present):

   ```bash
   python3 scripts/install_data.py ~/Downloads/wayup.zip
   ```

3. Sanity check: draw hold boxes on the first and last frame of every video.

   ```bash
   python3 scripts/check_alignment.py        # writes results/alignment/*.jpg
   ```

Resulting layout:

```
data/raw/wayup/
├── README.md, topo_green.png, topo_orange.png, participant_form_data.csv
└── p1, p2a, p2b, p3 … p10/                  # 11 recordings, 10 climbers (p2a, p2b = same person)
    └── {green,orange}.mp4, {green,orange}_holdUsage.csv,
        {green,orange}_routeAnnotation.json, {green,orange}_bolt_holes.json
```

### Team Drive layout

Shared folders: [`data/`](https://drive.google.com/drive/folders/14LDiwWxDYSHDEanNLgf-o8uWpW3GYr23) and [`deliverables/`](https://drive.google.com/drive/folders/1XWcjto1pSWkWLWmeqXtztriikkjzU9bY) (view access for course staff);
the [root folder](https://drive.google.com/drive/folders/141TIhRlO4kW7jZ-73BvQd77yGjOUokYU) is team-only.

```
HoldOn/
├── README.md                  # this layout and the rules below
├── data/                      # mirrors data/ in this repo
│   ├── raw/                   # wayup.zip (+ .sha256); later kilter.zip (+ .sha256)
│   ├── interim/pose2d/        # 2D keypoints, one folder per version
│   ├── interim/wham/          # WHAM outputs, one folder per version
│   └── processed/features/    # feature tables
├── checkpoints/               # trained Track 2 weights
├── results/                   # overlay videos, full failure gallery
├── deliverables/              # phase1_proposal/, phase2_progress/, phase3_final/, demo/
└── team/                      # raw phone clips, consent forms, notes (team only)
```

| Folder | Produced by |
| --- | --- |
| `data/raw/wayup.zip` | `scripts/package_wayup.py` |
| `data/raw/kilter.zip` | `scripts/convert_cfr.sh`, `scripts/label_kilter.py` |
| `data/interim/pose2d/`, `data/interim/wham/` | `external/pose2d/`, `external/wham/` |
| `data/processed/features/` | `holdon.features.build` |
| `checkpoints/` | `scripts/run_track2.py` |

Rules:

- **Versioned, never overwritten:** each model output goes in a new folder `<name>_<YYYY-MM-DD>/` with a
  `MANIFEST.md` (template: [docs/drive_manifest_template.md](docs/drive_manifest_template.md)) recording
  the model, config, code commit and input. Results in the report must name the version they used.
- **Archives carry a checksum:** every `.zip` has a `.sha256` next to it, mirrored in `configs/paths.yaml`.
- **Download, then install locally:** do not point `data/` at a Google Drive for desktop folder
  (on-demand sync makes frame-by-frame video reading slow and causes sync conflicts).

### How `wayup.zip` was made

From the original `dataset.zip` (Zenodo, DOI 10.5281/zenodo.15196866, 20.9 GB) we kept every file
unchanged except that we removed the 22 `*_large.mp4` files (4K / 50 fps originals; the labels refer
to the 25 fps videos).

Maintainers rebuild it with `python3 scripts/package_wayup.py` (output in `data/share/`), then update
`shared.wayup_zip_sha256` in `configs/paths.yaml` and upload the zip to the Drive. The archive is
deterministic: the same input files always give the same checksum.

**Participant data:** `participant_form_data.csv` contains participant demographics. It is shared with the team and course staff inside `wayup.zip`, but use it only for grouped analysis (e.g. by climbing experience): never report individual values or link them to people in videos or figures.

File formats and known quirks (participant `p2a`/`p2b`, labels without spaces, one unparseable row,
one file without a homography, end frames one past the video) are documented in
[docs/dataset_schema.md](docs/dataset_schema.md).

## Reproducing results

_TODO._

## Known issues

- The Way Up camera intrinsics are unknown; Track 3 uses an estimated focal length.
- SMPL body model files (needed by WHAM) are licensed for non-commercial research only and may not be
  redistributed; each member registers at smpl.is.tue.mpg.de and downloads them separately.

## License and attribution

- **Code:** MIT License, see [LICENSE](LICENSE).
- **The Way Up dataset:** © Maschek & Schedl, licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Our Drive copy is the unmodified processed
  subset (4K videos removed); all derived files (pose outputs, features, labels for Kilter) are our own.

```bibtex
@inproceedings{maschek2025wayup,
  title     = {The Way Up: A Dataset for Hold Usage Detection in Sport Climbing},
  author    = {Maschek, Anna and Schedl, David C.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2025},
  note      = {arXiv:2505.12854, dataset DOI 10.5281/zenodo.15196866}
}
```

- **Third-party models:** WHAM, SMPL, MMPose / ViTPose and Ultralytics YOLO keep their own licenses;
  we do not redistribute their weights.

## Team conventions

### Code
- Use `python3` and `pip3` in commands, scripts and docs, never `python` or `pip`.
- Code comments, docstrings, commit messages and documentation are in English.
- Python 3.10+, type hints on public functions, docstrings on every module and public function.
- Logic lives in the `src/holdon/` package; `scripts/` are thin entry points that call package code.
- Paths and hyperparameters go in `configs/`, not hard-coded in modules.
- Tests use `pytest`, run on CPU in seconds with tiny synthetic fixtures, and never need the real dataset or a GPU.

### Data
- Never commit datasets, videos, model weights or large outputs (`data/`, `checkpoints/`, `results/` are git-ignored).
- Use only the processed videos; skip `*_large.mp4`.
- Inspect real files and document their schema in `docs/` before writing a loader; do not guess field names.
- `participant_form_data.csv` contains participant demographics. It is shared with the team and course staff inside `wayup.zip`, but use it only for grouped analysis (e.g. by climbing experience): never report individual values or link them to people in videos or figures.
- Splits are leave-one-participant-out; never split by random frames.

### Architecture
- All tracks emit contact events in the shared format (`docs/event_format.md`) and are scored by `holdon.evaluation`.
- WHAM (and the 2D pose model) run in their own environment or Colab and write per-video files;
  the main package reads those files and never imports them.
- Prefer OpenCV-compatible implementations.

### Collaboration
- Each member commits from their own GitHub account, on feature branches, merged through pull requests.
- Small, focused commits with descriptive messages.
- Discuss with the team before adding heavy dependencies or changing the shared event format or evaluation protocol.

## Contributions

_TODO: per-member contributions._
