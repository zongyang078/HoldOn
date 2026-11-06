# WHAM (separate environment)

WHAM has heavy dependencies and needs a GPU, so it runs in its own environment or in Colab.
The main `holdon` package never imports WHAM; it reads the files written here.

- Mode: fixed camera, `--estimate_local_only`.
- Output: one `.npz` per video in `data/interim/wham/` (SMPL joints / vertices per frame).
- _TODO: setup steps, Colab link, exact `.npz` keys (record them here once fixed)._
