"""Tests for the Track 2 feature table (synthetic pose, two holds)."""

from __future__ import annotations

import numpy as np
import pytest

from holdon.features.build import build_video_table, concat, label_array, rolling
from tests.conftest import make_climb


def test_rolling_mean_min_with_nan_and_edges():
    x = np.array([1.0, np.nan, 3.0, 4.0, 10.0])
    assert rolling(x, 1, "mean").tolist() == [1.0, 2.0, 3.5, 17 / 3, 7.0]
    assert rolling(x, 1, "min").tolist() == [1.0, 1.0, 3.0, 3.0, 4.0]
    assert np.isnan(rolling(np.full(3, np.nan), 1, "min")).all()
    assert rolling(np.ones((4, 2)), 2, "mean").shape == (4, 2)


def test_label_array_ignores_wall_pushes():
    _, holds, labeled = make_climb("p1_green")
    lab = label_array(labeled, "left_hand", [h.hold_id for h in holds], 60)
    assert lab[:, 0].sum() == 30 and lab[:, 1].sum() == 0


def test_build_video_table_rows_labels_and_toggles(track2_cfg):
    pose, holds, labeled = make_climb("p1_green")
    table = build_video_table(pose, holds, labeled, track2_cfg)
    assert {"d_tip", "speed_tip", "d_tip_mean2", "speed_joint_min2", "rank"} <= set(table.names)
    assert table.X.shape == (len(table), len(table.names))
    hand0 = (table.limb == 0) & (table.hold == "0")
    assert set(table.frame[hand0 & table.label]) == set(range(10, 40))
    assert table.stats["hands"]["gt_frames"] == 30 and table.stats["hands"]["gt_frames_in_candidates"] == 30
    assert not table.label[table.limb != 0].any()  # other limbs hang far away: no candidates, no labels

    resting = hand0 & (table.frame >= 12) & (table.frame <= 37)  # away from arrival / departure
    moving = (table.limb == 0) & (table.frame >= 45)
    col = table.names.index("speed_tip")
    assert np.nanmax(table.X[resting, col]) < 0.005 < np.nanmedian(table.X[moving, col])

    cfg = {**track2_cfg, "features": {**track2_cfg["features"], "stillness": False, "context": False}}
    small = build_video_table(pose, holds, labeled, cfg)
    assert "speed_tip" not in small.names and "d_tip_mean2" not in small.names and len(small) == len(table)

    both = concat([table, table])
    assert len(both) == 2 * len(table) and both.stats["hands"]["gt_frames"] == 60


def test_distance3d_not_available(track2_cfg):
    pose, holds, labeled = make_climb("p1_green")
    cfg = {**track2_cfg, "features": {**track2_cfg["features"], "distance3d": True}}
    with pytest.raises(NotImplementedError):
        build_video_table(pose, holds, labeled, cfg)
