# MANIFEST — <output name>, e.g. rtmpose-wb_2026-11-15

Copy this file into every versioned output folder on the team Drive
(`data/interim/*/<name>_<YYYY-MM-DD>/`, `data/processed/*/…`, `checkpoints/…`).
Never overwrite an existing version folder; create a new one instead.

| Field | Value |
| --- | --- |
| Created by | <name> |
| Date | <YYYY-MM-DD> |
| Code commit | <git commit hash of this repo used to produce the files> |
| Producer | <script or notebook, e.g. external/wham/wham_inference.ipynb> |
| Model / weights | <model name, checkpoint file, version or URL> |
| Config | <config file and any non-default parameters> |
| Input | <e.g. wayup.zip sha256 d7215ab0…, all 22 videos> |
| Environment | <Colab GPU type / local, key package versions> |

## Files

<one line per file pattern, e.g. `<video_id>.npz`: per-frame SMPL joints (T×24×3, metres, camera frame)>

## Notes

<known problems, failed videos, anything a teammate must know before using these files>
