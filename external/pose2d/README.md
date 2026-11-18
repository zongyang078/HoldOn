# 2D pose (separate environment)

`run_pose2d.py` runs person detection + 2D pose on every processed The Way Up video and writes one
`.npz` per video. The main package reads these files with `holdon.data.external.load_pose2d`; it never
imports the pose libraries. File format: [docs/pose2d_format.md](../../docs/pose2d_format.md).

We avoid MMPose / mmcv (their builds rarely match current torch and CUDA). Both backends install with
`pip3` alone:

| `--model` | Weights | Keypoints | Used for |
| --- | --- | --- | --- |
| `rtmw` | RTMW-x 384×288 (RTMPose whole-body, OpenMMLab), ONNX via `rtmlib` | 133 COCO-WholeBody (body, feet, hands, face) | fingertip/toe variant, Track 2 features |
| `vitpose` | ViTPose++-H (`usyd-community/vitpose-plus-huge`), HuggingFace `transformers` | 17 COCO | reproducing The Way Up baseline (Track 1) |

Both use the same YOLOX-m person detector and the same climber selection (follow the previous frame's
box by IoU, else the largest box), so differences come from the pose model only.

## Colab (simplest)

Open [`pose2d_colab.ipynb`](https://colab.research.google.com/github/zongyang078/HoldOn/blob/feature/2d-line/external/pose2d/pose2d_colab.ipynb)
(change the branch in the URL to `main` once merged), pick a GPU runtime and *Run all*. It installs the
backends, copies `wayup.zip` from the team Drive and writes the outputs back to the Drive.

## Setup (GCP VM or any Linux machine with an NVIDIA GPU)

```bash
python3 -m venv ~/pose-env && source ~/pose-env/bin/activate
pip3 install -r external/pose2d/requirements.txt
pip3 uninstall -y onnxruntime && pip3 install onnxruntime-gpu==1.32.0
python3 -c "import onnxruntime as o; print(o.get_available_providers())"   # must list CUDAExecutionProvider
python3 -c "import torch; print(torch.cuda.is_available())"                 # must print True
```

Put the dataset at `data/raw/wayup/` (install `wayup.zip` as in the main README). Weights are
downloaded on first use (`~/.cache/rtmlib`, `~/.cache/huggingface`).

On macOS with the python.org installer, downloads fail with `CERTIFICATE_VERIFY_FAILED`; run
`export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")` first.

## Run

```bash
# smoke test: 50 frames of one video
python3 external/pose2d/run_pose2d.py --model rtmw --device cuda --only p1_green --max-frames 50 --out /tmp/pose_smoke

# full runs (all 22 videos, ~39k frames each)
python3 external/pose2d/run_pose2d.py --model rtmw --device cuda
python3 external/pose2d/run_pose2d.py --model vitpose --device cuda
```

Output goes to `data/interim/pose2d/<model>_<YYYY-MM-DD>/` (e.g. `rtmw-x-384_2026-11-19/`) with an
auto-filled `MANIFEST.md`; fill in "Created by" and "Notes" before uploading the folder to the team Drive.
An interrupted run resumes when restarted with the same `--date`: finished videos are skipped and
each file is written atomically. Run from a committed checkout so the manifest records a clean commit.

Measured on CPU (Apple M-series, for reference only): RTMW about 6 fps, ViTPose++-H about 1.5 fps.
