# 2D whole-body pose (separate environment)

MMPose (RTMPose-WholeBody) or ViTPose runs in its own environment because MMPose/mmcv
installs often conflict with the main PyTorch version.
The main package reads the exported keypoint files through `holdon.data.external`.

- Output: one file per video in `data/interim/pose2d/` (per-frame keypoints + scores).
- _TODO: environment setup, model checkpoint, exact output format._
