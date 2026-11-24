"""Assemble per-frame feature tables (video, frame, limb, hold, features) with feature toggles for ablation.

One row per candidate (frame, limb, hold): the holds within ``candidates.radius_px`` of the limb
that are also among its ``candidates.max_per_limb`` nearest holds (``holdon.features.overlap2d``).
This keeps the table small (a few rows per frame and limb instead of every hold) at the cost of
ground-truth contacts whose hold is not a candidate; those frames are counted in ``FeatureTable.stats``.

Feature groups (switched by ``features`` in ``configs/track2.yaml``):

- ``overlap2d``: fingertip/toe and wrist/ankle distance to the box (px), inside-box flags, rank of the
  hold among the frame's holds, number of holds within the radius, keypoint confidences, limb type
  (hand / foot) and hold type (foothold or not);
- ``stillness``: speed of the fingertip/toe centre and of the wrist/ankle (box heights per frame);
- ``context``: rolling mean and min over ``context_frames`` (±2 / ±6 / ±12) of the enabled distance and
  speed features;
- ``distance3d``: not available yet (needs ``versions.distance3d``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from holdon.data.external import Pose2D
from holdon.data.wayup import WALL_HOLD_ID, Hold, LabeledEvent
from holdon.events import LIMBS
from holdon.features import overlap2d, stillness
from holdon.models.rules import LIMB_KEYPOINTS, min_score_for


def rolling(x: np.ndarray, half: int, how: str) -> np.ndarray:
    """NaN-aware centred rolling ``mean`` or ``min`` over ``2 * half + 1`` frames along axis 0.

    Works on (T,) or (T, H) arrays; windows are truncated at the ends; NaN only if the whole
    window is NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    if half <= 0:
        return x.copy()
    pad = [(half, half)] + [(0, 0)] * (x.ndim - 1)
    win = sliding_window_view(np.pad(x, pad, constant_values=np.nan), 2 * half + 1, axis=0)
    valid = ~np.isnan(win)
    n = valid.sum(axis=-1)
    if how == "mean":
        out = np.where(valid, win, 0.0).sum(axis=-1) / np.maximum(n, 1)
    elif how == "min":
        out = np.where(valid, win, np.inf).min(axis=-1)
    else:
        raise ValueError(f"Unknown rolling statistic {how!r}")
    return np.where(n > 0, out, np.nan)


@dataclass
class FeatureTable:
    """Candidate rows of one or more videos.

    ``limb`` indexes ``holdon.events.LIMBS``; ``hold`` is the hold id string; ``label`` is True when
    the ground truth has this limb on this hold in this frame (wall pushes ``w`` are not labels).
    """

    names: tuple[str, ...]
    X: np.ndarray  # (N, F) float32, NaN = unknown (imputed by the model pipeline)
    video: np.ndarray  # (N,) str
    frame: np.ndarray  # (N,) int32
    limb: np.ndarray  # (N,) int8
    hold: np.ndarray  # (N,) str
    label: np.ndarray  # (N,) bool
    stats: dict[str, dict[str, int]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.label)

    def select(self, mask: np.ndarray) -> "FeatureTable":
        """Rows where ``mask`` is True (stats are kept unchanged)."""
        return FeatureTable(self.names, self.X[mask], self.video[mask], self.frame[mask], self.limb[mask],
                            self.hold[mask], self.label[mask], self.stats)


def concat(tables: Sequence[FeatureTable]) -> FeatureTable:
    """Stack tables with the same feature names; candidate stats are summed."""
    if not tables:
        raise ValueError("No tables to concatenate")
    names = tables[0].names
    if any(t.names != names for t in tables):
        raise ValueError("Feature names differ between tables")
    stats: dict[str, dict[str, int]] = {}
    for t in tables:
        for group, counts in t.stats.items():
            for k, v in counts.items():
                stats.setdefault(group, {}).setdefault(k, 0)
                stats[group][k] += v
    cat = lambda attr: np.concatenate([getattr(t, attr) for t in tables])  # noqa: E731
    return FeatureTable(names, cat("X"), cat("video"), cat("frame"), cat("limb"), cat("hold"), cat("label"), stats)


def label_array(labeled: Sequence[LabeledEvent], limb: str, hold_ids: Sequence[str], num_frames: int) -> np.ndarray:
    """(T, H) ground-truth contact mask of one limb (``w`` events ignored)."""
    index = {h: i for i, h in enumerate(hold_ids)}
    out = np.zeros((num_frames, len(hold_ids)), bool)
    for le in labeled:
        e = le.event
        if e.limb != limb or e.hold_id == WALL_HOLD_ID or e.hold_id not in index:
            continue
        out[e.start_frame : min(e.end_frame, num_frames - 1) + 1, index[e.hold_id]] = True
    return out


def build_video_table(
    pose: Pose2D, holds: Sequence[Hold], labeled: Sequence[LabeledEvent], cfg: Mapping[str, Any],
) -> FeatureTable:
    """Feature table of one video. ``cfg`` is ``configs/track2.yaml``; ``labeled`` may be empty."""
    feats = cfg["features"]
    if feats.get("distance3d"):
        raise NotImplementedError("distance3d features need versions.distance3d; set features.distance3d: false")
    cand = cfg["candidates"]
    radius, k = float(cand["radius_px"]), int(cand["max_per_limb"])
    windows = [int(w) for w in cfg.get("context_frames", [])] if feats.get("context") else []
    min_score = min_score_for(pose.model, cfg["min_score"])
    boxes = overlap2d.hold_boxes(holds)
    centres = overlap2d.box_centres(boxes)
    hold_ids = np.array([h.hold_id for h in holds])
    foothold = np.array([h.is_foothold for h in holds], dtype=np.float64)
    scale = stillness.body_scale(pose.bbox)
    T, H = pose.num_frames, len(holds)
    tips_set, joint_set = LIMB_KEYPOINTS[cfg["keypoints"]["tips"]], LIMB_KEYPOINTS[cfg["keypoints"]["joint"]]

    rows: list[dict[str, np.ndarray]] = []
    meta = {"frame": [], "limb": [], "hold": [], "label": []}
    stats: dict[str, dict[str, int]] = {}
    for li, limb in enumerate(LIMBS):
        tips, tip_score = overlap2d.smoothed_tracks(pose, tips_set[limb], min_score, cfg["smoothing"])
        joint, joint_score = overlap2d.smoothed_tracks(pose, joint_set[limb], min_score, cfg["smoothing"])
        d_tip = overlap2d.min_distance(tips, boxes)
        d_joint = overlap2d.min_distance(joint, boxes)
        centre = overlap2d.mean_point(tips)
        tiebreak = np.linalg.norm(centre[:, None, :] - centres[None], axis=-1)
        # candidates: by the closer of the two keypoint sets, so either can bring a hold in
        d_any = np.fmin(d_tip, d_joint)
        rank = overlap2d.distance_rank(d_any, tiebreak)
        mask = overlap2d.candidate_mask(d_any, rank, radius, k)
        labels = label_array(labeled, limb, hold_ids, T)

        group = "hands" if limb.endswith("hand") else "feet"
        s = stats.setdefault(group, {"gt_frames": 0, "gt_frames_in_candidates": 0, "gt_frames_no_pose": 0})
        s["gt_frames"] += int(labels.sum())
        s["gt_frames_in_candidates"] += int((labels & mask).sum())
        s["gt_frames_no_pose"] += int((labels & np.isnan(d_any).all(axis=1, keepdims=True)).sum())

        t_idx, h_idx = np.nonzero(mask)
        cols: dict[str, np.ndarray] = {}
        if feats.get("overlap2d", True):
            cols["d_tip"] = d_tip[t_idx, h_idx]
            cols["d_joint"] = d_joint[t_idx, h_idx]
            cols["inside_tip"] = (d_tip[t_idx, h_idx] == 0).astype(np.float64)
            cols["inside_joint"] = (d_joint[t_idx, h_idx] == 0).astype(np.float64)
            cols["rank"] = rank[t_idx, h_idx].astype(np.float64)
            cols["n_within"] = overlap2d.count_within(d_any, radius)[t_idx].astype(np.float64)
            cols["score_tip"] = tip_score[t_idx]
            cols["score_joint"] = joint_score[t_idx]
            cols["is_hand"] = np.full(len(t_idx), float(limb.endswith("hand")))
            cols["is_foothold"] = foothold[h_idx]
            for w in windows:
                for name, arr in (("d_tip", d_tip), ("d_joint", d_joint)):
                    for how in ("mean", "min"):
                        cols[f"{name}_{how}{w}"] = rolling(arr, w, how)[t_idx, h_idx]
        if feats.get("stillness", True):
            speeds = {"speed_tip": stillness.normalized_speed(centre, scale),
                      "speed_joint": stillness.normalized_speed(overlap2d.mean_point(joint), scale)}
            for name, sp in speeds.items():
                cols[name] = sp[t_idx]
                for w in windows:
                    for how in ("mean", "min"):
                        cols[f"{name}_{how}{w}"] = rolling(sp, w, how)[t_idx]
        if not cols:
            raise ValueError("All feature groups are disabled")
        rows.append(cols)
        meta["frame"].append(t_idx.astype(np.int32))
        meta["limb"].append(np.full(len(t_idx), li, np.int8))
        meta["hold"].append(hold_ids[h_idx])
        meta["label"].append(labels[t_idx, h_idx])

    names = tuple(rows[0])
    X = np.concatenate([np.column_stack([r[n] for n in names]) for r in rows]).astype(np.float32)
    n = len(X)
    return FeatureTable(names, X, np.full(n, pose.video_id, dtype=object), np.concatenate(meta["frame"]),
                        np.concatenate(meta["limb"]), np.concatenate(meta["hold"]).astype(object),
                        np.concatenate(meta["label"]), stats)
