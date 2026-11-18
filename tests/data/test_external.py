"""Tests for reading exported 2D pose files, using tiny synthetic .npz files."""

from __future__ import annotations

import numpy as np
import pytest

from holdon.data.external import load_pose2d, pose2d_path

COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def _write(path, t=4, names=COCO17, keypoint_set="coco17", version=2, kpt_shape=None):
    k = len(names)
    keypoints = np.arange(t * k * 2, dtype=np.float32).reshape(t, k, 2)
    keypoints[2] = np.nan  # frame 2: no person
    scores = np.full((t, k), 0.9, dtype=np.float32)
    scores[2] = 0
    bbox = np.tile(np.array([10, 20, 110, 220], dtype=np.float32), (t, 1))
    bbox[2] = np.nan
    np.savez_compressed(
        path,
        keypoints=keypoints if kpt_shape is None else np.zeros(kpt_shape, np.float32),
        scores=scores,
        bbox=bbox,
        num_persons=np.array([1, 2, 0, 1], dtype=np.int16)[:t],
        all_boxes=np.full((t, 8, 4), np.nan, dtype=np.float32),
        route_region=np.array([0, 0, 720, 1280], dtype=np.float32),
        fps=np.float32(25),
        width=np.int32(720),
        height=np.int32(1280),
        source_num_frames=np.int32(t),
        format_version=np.int32(version),
        video_id=np.str_("p0_green"),
        model=np.str_("vitpose-test"),
        detector=np.str_("yolox"),
        keypoint_set=np.str_(keypoint_set),
        keypoint_names=np.array(names),
    )
    return path


def test_load_pose2d_roundtrip(tmp_path):
    pose = load_pose2d(_write(pose2d_path(tmp_path, "p0_green")))
    assert (pose.video_id, pose.model, pose.keypoint_set) == ("p0_green", "vitpose-test", "coco17")
    assert pose.num_frames == 4 and (pose.width, pose.height, pose.fps) == (720, 1280, 25.0)
    xy, score = pose.track("left_wrist")
    assert xy.shape == (4, 2) and score.shape == (4,)
    assert xy[0].tolist() == [18.0, 19.0]  # column 9 of frame 0
    assert np.isnan(xy[2]).all() and score[2] == 0
    assert pose.num_persons.tolist() == [1, 2, 0, 1]
    assert pose.all_boxes.shape == (4, 8, 4) and pose.route_region.tolist() == [0, 0, 720, 1280]


def test_unknown_keypoint_name(tmp_path):
    pose = load_pose2d(_write(tmp_path / "a.npz"))
    with pytest.raises(KeyError):
        pose.index("left_big_toe")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"version": 1}, "format_version"),
        ({"keypoint_set": "halpe26"}, "unknown keypoint_set"),
        ({"names": COCO17[:16]}, "keypoint names"),
        ({"kpt_shape": (4, 17, 3)}, "keypoints has shape"),
    ],
)
def test_invalid_files_rejected(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        load_pose2d(_write(tmp_path / "bad.npz", **kwargs))
