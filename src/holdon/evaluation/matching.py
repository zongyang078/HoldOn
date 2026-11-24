"""Event matching of docs/evaluation_protocol.md: temporal IoU and greedy one-to-one matching.

A prediction can match only a ground-truth event with the same limb and the same hold. Matching is
greedy per video: predictions in descending score each take the unused eligible ground-truth event
with the highest tIoU; the pair counts if it passes the threshold. Threshold 0.0 means tIoU > 0,
any other t means tIoU >= t.

Moved unchanged from ``holdon.models.track1_tuning`` (the Phase 2 reference implementation).
"""

from __future__ import annotations

from typing import Sequence

from holdon.events import ContactEvent


def tiou(a: ContactEvent, b: ContactEvent) -> float:
    """Temporal IoU of two inclusive frame intervals."""
    inter = min(a.end_frame, b.end_frame) - max(a.start_frame, b.start_frame) + 1
    if inter <= 0:
        return 0.0
    union = max(a.end_frame, b.end_frame) - min(a.start_frame, b.start_frame) + 1
    return inter / union


def _passes(value: float, threshold: float) -> bool:
    return value > 0 if threshold == 0 else value >= threshold


def match_events(
    pred: Sequence[ContactEvent], gt: Sequence[ContactEvent], threshold: float
) -> list[tuple[int, int, float]]:
    """Greedy matching; returns ``(pred index, gt index, tIoU)`` for each matched pair."""
    order = sorted(range(len(pred)), key=lambda i: -pred[i].score)
    used: set[int] = set()
    pairs = []
    for i in order:
        p = pred[i]
        best, best_iou = -1, 0.0
        for j, g in enumerate(gt):
            if j in used or g.limb != p.limb or g.hold_id != p.hold_id:
                continue
            value = tiou(p, g)
            if value > best_iou:
                best, best_iou = j, value
        if best >= 0 and _passes(best_iou, threshold):
            used.add(best)
            pairs.append((i, best, best_iou))
    return pairs
