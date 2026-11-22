"""Grouped analysis: hands vs feet, route, occluded vs visible, climber experience.

Definitions are the "Breakdowns" of docs/evaluation_protocol.md:

- **hands / feet:** predictions and ground truth are both filtered to the two hand (foot) limbs,
  then scored with ``holdon.evaluation.metrics.evaluate``.
- **route:** macro average over the videos of one route (11 per route on The Way Up).
- **occluded / visible:** frame recall. Ground-truth contact frames inside (resp. outside) the
  event's ``framesOccluded`` ranges; a frame is recalled if any prediction with the same limb and
  hold covers it. Frames of all events of all videos are pooled.
- **climber experience:** videos grouped by the participant's answer to
  ``For how long have you been climbing?`` in ``participant_form_data.csv``; only group averages are
  returned. Groups with fewer than ``min_participants`` participants are merged into the
  neighbouring category so that no result belongs to a single person.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from holdon.data.wayup import LabeledEvent
from holdon.evaluation.metrics import DEFAULT_THRESHOLDS, FEET, HANDS, GroundTruth, evaluate
from holdon.events import ContactEvent

EXPERIENCE_COLUMN = "For how long have you been climbing?"
# Answer options in increasing order of experience (the form's categories).
EXPERIENCE_ORDER: tuple[str, ...] = ("less than 1 year", "1-2 years", "5-10 years", "more than 10 years")

Metrics = dict[float, dict[str, float]]


def route_of(video_id: str) -> str:
    """Route of a video id, e.g. ``p2b_orange`` -> ``orange``."""
    return video_id.rsplit("_", 1)[1]


def recording_of(video_id: str) -> str:
    """Recording of a video id, e.g. ``p2b_orange`` -> ``p2b``."""
    return video_id.rsplit("_", 1)[0]


def _subset(by_video: Mapping[str, Sequence], videos: Sequence[str]) -> dict[str, Sequence]:
    return {v: by_video[v] for v in videos if v in by_video}


def by_limb_type(
    pred_by_video: Mapping[str, Sequence[ContactEvent]],
    gt_by_video: Mapping[str, Sequence[GroundTruth]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Metrics]:
    """``{"all", "hands", "feet"}`` -> metrics; hands/feet filter both sides before matching."""
    return {
        "all": evaluate(pred_by_video, gt_by_video, thresholds),
        "hands": evaluate(pred_by_video, gt_by_video, thresholds, HANDS),
        "feet": evaluate(pred_by_video, gt_by_video, thresholds, FEET),
    }


def by_video_group(
    pred_by_video: Mapping[str, Sequence[ContactEvent]],
    gt_by_video: Mapping[str, Sequence[GroundTruth]],
    group_of: Callable[[str], str | None],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    limbs: Sequence[str] | None = None,
) -> dict[str, Metrics]:
    """Macro metrics within each group of videos (``group_of(video) -> group``; None = left out)."""
    groups: dict[str, list[str]] = {}
    for v in gt_by_video:
        g = group_of(v)
        if g is not None:
            groups.setdefault(g, []).append(v)
    return {
        g: evaluate(_subset(pred_by_video, vids), _subset(gt_by_video, vids), thresholds, limbs)
        for g, vids in groups.items()
    }


def by_route(
    pred_by_video: Mapping[str, Sequence[ContactEvent]],
    gt_by_video: Mapping[str, Sequence[GroundTruth]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Metrics]:
    """Macro metrics per route (``green``, ``orange``)."""
    return dict(sorted(by_video_group(pred_by_video, gt_by_video, route_of, thresholds).items()))


# ----------------------------------------------------------------------------- occlusion


@dataclass(frozen=True)
class FrameRecall:
    """Pooled frame recall of one subset of ground-truth contact frames."""

    frames: int
    covered: int

    @property
    def recall(self) -> float:
        """Share of frames covered by a matching-limb, matching-hold prediction (0 if no frames)."""
        return self.covered / self.frames if self.frames else 0.0


def occlusion_frame_recall(
    pred_by_video: Mapping[str, Sequence[ContactEvent]],
    labeled_by_video: Mapping[str, Sequence[LabeledEvent]],
    limbs: Sequence[str] | None = None,
) -> dict[str, FrameRecall]:
    """Frame recall of occluded and visible ground-truth contact frames.

    For every ground-truth event, its frames are split into occluded (inside one of its
    ``framesOccluded`` ranges) and visible (the rest). A frame counts as recalled if any prediction
    with the same limb and the same hold covers it. Counts are pooled over all events of all videos.

    Returns:
        ``{"occluded": FrameRecall, "visible": FrameRecall}``.
    """
    totals = {"occluded": [0, 0], "visible": [0, 0]}
    for vid, labeled in labeled_by_video.items():
        preds = pred_by_video.get(vid, [])
        for le in labeled:
            g = le.event
            if limbs is not None and g.limb not in limbs:
                continue
            n = g.end_frame - g.start_frame + 1
            occluded = np.zeros(n, bool)
            for a, b in le.occluded:
                occluded[max(a, g.start_frame) - g.start_frame : min(b, g.end_frame) - g.start_frame + 1] = True
            covered = np.zeros(n, bool)
            for p in preds:
                if p.limb != g.limb or p.hold_id != g.hold_id:
                    continue
                a, b = max(p.start_frame, g.start_frame), min(p.end_frame, g.end_frame)
                if a <= b:
                    covered[a - g.start_frame : b - g.start_frame + 1] = True
            for name, sel in (("occluded", occluded), ("visible", ~occluded)):
                totals[name][0] += int(sel.sum())
                totals[name][1] += int((covered & sel).sum())
    return {name: FrameRecall(frames, hit) for name, (frames, hit) in totals.items()}


# ----------------------------------------------------------------------------- experience


def load_experience(form_path: str | Path, column: str = EXPERIENCE_COLUMN) -> dict[int, str]:
    """Participant number -> experience answer, read from ``participant_form_data.csv``.

    Internal input of ``experience_groups``; never report this mapping itself.
    """
    with Path(form_path).open(newline="", encoding="utf-8") as fh:
        return {int(row["Participant ID"]): row[column].strip() for row in csv.DictReader(fh)}


def experience_groups(
    experience: Mapping[int, str],
    min_participants: int = 2,
    order: Sequence[str] = EXPERIENCE_ORDER,
) -> dict[int, str]:
    """Participant -> group label, merging categories with fewer than ``min_participants``.

    Categories are walked in ``order`` (unknown answers last, alphabetically); a category that is
    too small is merged into the next one (the last one into the previous one). Merged labels join
    the answers with ``" + "``, e.g. ``"less than 1 year + 1-2 years"``.
    """
    rank = {c: i for i, c in enumerate(order)}
    cats = sorted(set(experience.values()), key=lambda c: (rank.get(c, len(rank)), c))
    groups = [[c] for c in cats]

    def size(grp: list[str]) -> int:
        return sum(1 for v in experience.values() if v in grp)

    while len(groups) > 1:
        small = next((i for i, grp in enumerate(groups) if size(grp) < min_participants), None)
        if small is None:
            break
        lo, hi = (small, small + 1) if small + 1 < len(groups) else (small - 1, small)
        groups[lo : hi + 1] = [groups[lo] + groups[hi]]
    label = {c: " + ".join(grp) for grp in groups for c in grp}
    return {p: label[c] for p, c in experience.items()}


def by_experience(
    pred_by_video: Mapping[str, Sequence[ContactEvent]],
    gt_by_video: Mapping[str, Sequence[GroundTruth]],
    group_of_participant: Mapping[int, str],
    participant_of: Callable[[str], int],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, tuple[int, int, Metrics]]:
    """Group label -> ``(participants, videos, macro metrics)`` over the group's videos.

    ``participant_of`` maps a recording (``p2b``) to its participant number. Only group averages are
    returned, never a per-person value; use ``experience_groups`` to avoid single-person groups.
    """
    def group_of(v: str) -> str | None:
        return group_of_participant.get(participant_of(recording_of(v)))

    metrics = by_video_group(pred_by_video, gt_by_video, group_of, thresholds)
    out = {}
    for g in sorted(metrics, key=lambda g: _group_rank(g)):
        vids = [v for v in gt_by_video if group_of(v) == g]
        people = {participant_of(recording_of(v)) for v in vids}
        out[g] = (len(people), len(vids), metrics[g])
    return out


def _group_rank(label: str) -> tuple[int, str]:
    first = label.split(" + ")[0]
    return (EXPERIENCE_ORDER.index(first) if first in EXPERIENCE_ORDER else len(EXPERIENCE_ORDER), label)


__all__ = [
    "FrameRecall", "by_experience", "by_limb_type", "by_route", "by_video_group", "experience_groups",
    "load_experience", "occlusion_frame_recall", "recording_of", "route_of",
]
