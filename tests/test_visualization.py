"""Smoke tests for overlay drawing and timeline plots."""

from __future__ import annotations

import numpy as np

from holdon.data.wayup import Hold
from holdon.events import ContactEvent
from holdon.visualization.overlay import contact_panel, draw_keypoints, highlight_holds
from holdon.visualization.timeline import plot_timeline


def test_draw_keypoints_and_highlight():
    frame = np.zeros((40, 40, 3), np.uint8)
    out = draw_keypoints(frame, [(10, 10), (np.nan, 5)], color=(0, 0, 255), radius=2)
    assert out[10, 10].tolist() == [0, 0, 255] and frame.sum() == 0  # input untouched
    out = highlight_holds(frame, [Hold("1", 5, 5, 30, 30, False), Hold("2", 0, 0, 3, 3, True)], ["1"],
                          (0, 255, 0), thickness=1)
    assert out[5, 20].tolist() == [0, 255, 0] and out[0, 0].tolist() == [0, 0, 0]


def test_plot_timeline(tmp_path):
    gt = [ContactEvent("v", "left_hand", "3", 0, 49)]
    pred = [ContactEvent("v", "left_hand", "3", 5, 60, 0.8)]
    fig = plot_timeline(gt, pred, fps=25, title="v")
    fig.savefig(tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 0


def test_contact_panel_colors_rows():
    band = contact_panel(300, {"left_hand": "3"}, {"left_hand": "3", "right_foot": "7"}, header="v")
    assert band.shape[1] == 300 and band.shape[0] > 100
    rows = band[:, :, 1] < 200  # dark pixels
    assert rows.any()
