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
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0   # OpenCV needs libGL on server images
python3 -m venv ~/pose-env && source ~/pose-env/bin/activate   # Python 3.12 recommended (uv venv -p 3.12)
pip3 install -r external/pose2d/requirements.txt
pip3 uninstall -y onnxruntime onnxruntime-gpu   # rtmlib pulls in the CPU build, which shadows the GPU one
pip3 install "onnxruntime-gpu[cuda,cudnn]==1.32.0"   # bundles CUDA 13 + cuDNN 9 libraries
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

Measured speed (frames per second, including video decoding and detection):

| Hardware | RTMW-x 384 | ViTPose++-H | All 22 videos, both models |
| --- | --- | --- | --- |
| GCP n1-standard-4 + T4 | 14 | 5 | about 2 h 45 min |
| Apple M-series CPU | 6 | 1.5 | not practical |

## Known issues

- **Slow RTMW download on cloud VMs:** `download.openmmlab.com` can be as slow as 600 kB/s from GCP
  (the 230 MB model takes several minutes and looks like a hang). Copy `~/.cache/rtmlib/hub/checkpoints/*.onnx`
  from a machine that already has them instead.
- **macOS tarballs:** a `tar` made on a Mac adds `._*` metadata files next to every video; delete them
  (`find data/raw/wayup -name "._*" -delete`). The script ignores them anyway, but they confuse file counts.
- **onnxruntime on CPU by accident:** `rtmlib` installs the CPU `onnxruntime`, which shadows
  `onnxruntime-gpu`; the script refuses to run with `--device cuda` in that case (see Setup).
