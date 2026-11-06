"""Tests for The Way Up loaders using tiny synthetic files that mimic the real format."""

from __future__ import annotations

import json

import numpy as np
import pytest

from holdon.data import wayup

CSV_TEXT = (
    "p2b,startFrame,endFrame,framesOccluded\n"
    'rh 2,47,239,"59-69, 133-195"\n'
    "lh2,73,166,73-78\n"
    "rf w,100,194,\n"
    "lf (wrong hold),854,864,\n"
    "lf 0,155,400,390-400\n"
)


@pytest.fixture
def recording_dir(tmp_path):
    rec = tmp_path / "p2b"
    rec.mkdir()
    (rec / "green_holdUsage.csv").write_text(CSV_TEXT)
    holds = {"holds": [
        {"topLeftCorner": {"x": 1, "y": 2}, "bottomRightCorner": {"x": 30, "y": 40}, "isFoothold": True},
        {"topLeftCorner": {"x": 5, "y": 6}, "bottomRightCorner": {"x": 7, "y": 8}, "isFoothold": False},
    ]}
    (rec / "green_routeAnnotation.json").write_text(json.dumps(holds))
    bolts = {
        "img_coords": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "world_coords": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        "homography_mat": np.eye(3).tolist(),
    }
    (rec / "green_bolt_holes.json").write_text(json.dumps(bolts))
    (tmp_path / "participant_form_data.csv").write_text("not a recording\n")
    return tmp_path


@pytest.mark.parametrize("label,expected", [
    ("rh 2", ("right_hand", "2")),
    ("lh2", ("left_hand", "2")),
    ("lf w", ("left_foot", "w")),
    (" rf 12 ", ("right_foot", "12")),
    ("lf (wrong hold)", None),
])
def test_parse_label(label, expected):
    assert wayup.parse_label(label) == expected


def test_parse_occlusion():
    assert wayup.parse_occlusion("59-69, 133-195, 7") == ((59, 69), (133, 195), (7, 7))
    assert wayup.parse_occlusion("") == ()


def test_participant_of():
    assert wayup.participant_of("p2a") == wayup.participant_of("p2b") == 2
    assert wayup.participant_of("p10") == 10
    with pytest.raises(ValueError):
        wayup.participant_of("dataset")


def test_load_hold_usage_skips_bad_label_and_clips(recording_dir, caplog):
    events = wayup.load_hold_usage(
        recording_dir / "p2b" / "green_holdUsage.csv", "p2b_green", num_frames=396
    )
    assert len(events) == 4
    first = events[0]
    assert first.event.limb == "right_hand" and first.event.hold_id == "2"
    assert first.occluded == ((59, 69), (133, 195))
    assert events[2].event.hold_id == wayup.WALL_HOLD_ID
    last = events[-1]
    assert last.event.end_frame == 395
    assert last.occluded == ((390, 395),)
    assert "wrong hold" in caplog.text


def test_load_recording(recording_dir):
    rec = wayup.load_recording(recording_dir, "p2b", "green")
    assert rec.video_id == "p2b_green" and rec.participant == 2
    assert [h.hold_id for h in rec.holds] == ["0", "1"]
    assert rec.holds[0].is_foothold and (rec.holds[0].x2, rec.holds[0].y2) == (30, 40)
    assert rec.wall.homography.shape == (3, 3)


def test_wall_reference_estimates_missing_homography(tmp_path):
    rng = np.random.default_rng(0)
    world = np.column_stack([rng.uniform(0, 135, 20), rng.uniform(0, 510, 20), np.zeros(20)])
    true_h = np.array([[0.6, 0.0, -160.0], [0.0, -0.6, 560.0], [0.0, 0.0, 1.0]])
    img = (np.linalg.inv(true_h) @ np.column_stack([world[:, :2], np.ones(20)]).T).T
    img = img[:, :2] / img[:, 2:]
    path = tmp_path / "bolts.json"
    path.write_text(json.dumps({"img_coords": img.tolist(), "world_coords": world.tolist()}))
    wall = wayup.load_wall_reference(path)
    np.testing.assert_allclose(wall.homography / wall.homography[2, 2], true_h, rtol=1e-5, atol=1e-6)


def test_list_recordings(recording_dir):
    assert wayup.list_recordings(recording_dir) == ["p2b"]
