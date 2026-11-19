"""Contact timeline plots (predicted vs ground truth) with matplotlib.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # file output only; works without a display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from holdon.events import LIMBS, ContactEvent  # noqa: E402

GT_COLOR = "#f0a020"
PRED_COLOR = "#c020c0"


def plot_timeline(
    gt: Sequence[ContactEvent],
    pred: Sequence[ContactEvent],
    fps: float,
    title: str = "",
    limbs: Sequence[str] = LIMBS,
) -> Figure:
    """One row per limb: ground-truth events above, predictions below, hold id written on each bar.

    Time axis in seconds. Returns the figure; the caller saves it.
    """
    fig, ax = plt.subplots(figsize=(12, 0.9 * len(limbs) + 1))
    for row, limb in enumerate(limbs):
        y = len(limbs) - 1 - row
        for events, offset, color in ((gt, 0.18, GT_COLOR), (pred, -0.18, PRED_COLOR)):
            for e in events:
                if e.limb != limb:
                    continue
                start, width = e.start_frame / fps, e.num_frames / fps
                ax.barh(y + offset, width, left=start, height=0.32, color=color, edgecolor="black", linewidth=0.3)
                if width > 1.2:
                    ax.text(start + width / 2, y + offset, e.hold_id, ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(limbs)), [limb.replace("_", " ") for limb in reversed(limbs)])
    ax.set_xlabel("time (s)")
    ax.set_title(title)
    handles = [Patch(color=GT_COLOR, label="ground truth"), Patch(color=PRED_COLOR, label="prediction")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8, ncols=2, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
