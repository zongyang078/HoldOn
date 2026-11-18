# WHAM (separate environment, 3D line)

WHAM has heavy dependencies and needs a GPU, so it runs in its own environment or in Colab.
The main `holdon` package never imports WHAM.

## What the 3D line must deliver

The **distance files** specified in [docs/interfaces.md](../../docs/interfaces.md) section 6:
`data/processed/distance3d/<name>_<YYYY-MM-DD>/<video_id>.npz`, all 22 videos, checked with
`holdon.data.external.load_distance3d`, plus `MANIFEST.md`; then set `versions.distance3d` in
`configs/paths.yaml`. Everything below is internal to the 3D line.

## Raw WHAM outputs (internal)

- Mode: fixed camera, `--estimate_local_only`.
- Input: the processed videos `data/raw/wayup/<recording>/{green,orange}.mp4` (25 fps); frame `t` of the
  output must be frame `t` of the video (check the count against `holdon.data.video.video_info`).
- Output: `data/interim/wham/<name>_<YYYY-MM-DD>/<video_id>.npz` + `MANIFEST.md`.
- SMPL model files: licensed per person (register at smpl.is.tue.mpg.de); never commit or share them.

_To fill in by the 3D line: setup steps or Colab link, the exact `.npz` keys and shapes, the chosen limb
points and hold model, the focal-length estimate._
