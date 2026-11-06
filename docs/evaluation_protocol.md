# Evaluation protocol

Summary; the authoritative description is section 3 of `PROJECT_CONTEXT.md`.

1. A prediction can match a ground-truth event only with the same `limb` and `hold_id`.
2. Temporal IoU on inclusive frame intervals: `tIoU = |Tp ∩ Tg| / |Tp ∪ Tg|`.
3. Greedy matching in descending prediction score; each event is used at most once.
4. TP = matched prediction, FP = unmatched prediction, FN = unmatched ground truth.
5. Precision, Recall, F1 at tIoU > 0, >= 0.3, >= 0.5, plus mean tIoU over matches.
6. Macro-average over videos.
7. Breakdowns: hands vs feet, route, occluded vs visible, climber experience.
8. Splits: leave-one-participant-out; thresholds tuned on training folds only.
