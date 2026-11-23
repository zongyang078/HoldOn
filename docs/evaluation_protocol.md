# Evaluation protocol

Shared by all three tracks. Implementation: `holdon.evaluation` and `scripts/evaluate.py`
(see [interfaces.md](interfaces.md) section 5). Every rule below is fixed.

## Matching

1. A prediction can match a ground-truth event only with the **same `limb` and same `hold_id`**.
2. Temporal IoU on inclusive frame intervals: `tIoU = |Tp ∩ Tg| / |Tp ∪ Tg|`
   (e.g. `[0, 9]` vs `[5, 14]` → 5 / 15).
3. Greedy matching per video: predictions in descending `score`; each takes the unused eligible
   ground-truth event with the highest tIoU; the pair counts if it passes the threshold. Each event is
   used at most once. Matching is redone for every threshold.
4. Thresholds (`configs/eval.yaml`): `0.0` means tIoU **> 0** (any overlap); `0.3` and `0.5` mean tIoU **≥** t.
5. TP = matched prediction, FP = unmatched prediction, FN = unmatched ground truth. No true negatives.

## Metrics

- Per video: `P = TP / (TP + FP)`, `R = TP / (TP + FN)`, `F1 = 2PR / (P + R)`; any of them is **0 when its
  denominator is 0** (e.g. a video without predictions has P = 0, not "undefined").
- **Macro average over all 22 videos** (each video weighs the same, no video is skipped).
- Mean tIoU: average tIoU over all matched pairs of all videos at the given threshold.
- Main number for comparisons: F1 at each threshold; the tuning objective is their mean (`mean_f1`).

## Ground truth

- All labeled events count, including the 31 wall pushes (`hold_id = "w"`: 4 hands, 27 feet). No track
  predicts them, so they are FN for everyone; report this in the text.
- The unparseable `lf (wrong hold)` row (p4 green) is skipped by the loader; 939 events remain.

## Breakdowns

| Breakdown | Definition |
| --- | --- |
| Hands / feet | filter **both** predictions and ground truth to the two hand (or foot) limbs, then score as above |
| Route | macro average over the 11 videos of one route |
| Occluded / visible | frame recall: share of ground-truth contact frames inside (resp. outside) the event's `framesOccluded` ranges that are covered by any prediction with the same limb and hold |
| Climber experience | group videos by `For how long have you been climbing?` in `participant_form_data.csv` (4 categories); report group averages only, never per person. A category with fewer than 2 participants is merged with the neighbouring one (only 1 participant answered `less than 1 year`, so it is reported together with `1-2 years`; `configs/eval.yaml` → `experience.min_participants`) |

## Tuning and splits

- Leave-one-participant-out: 10 folds, `p2a` + `p2b` held out together.
- Hyperparameters and thresholds are chosen **on the training participants of each fold only**, then
  applied to the held-out participant; reported numbers use only held-out predictions.
- Report the version folders used (`configs/paths.yaml` → `versions`).

## Comparing with the paper

The Way Up reports "accuracy" for its ViTPose baseline (~86.6% at tIoU > 0, ~69% at tIoU > 0.5) with
a different definition than our F1; quote it as context, not as a like-for-like number.
