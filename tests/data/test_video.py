"""Tests for OpenCV video helpers on a tiny synthetic video."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from holdon.data.video import iter_frames, read_frame, video_info


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (32, 24))
    for i in range(5):
        writer.write(np.full((24, 32, 3), i * 50, dtype=np.uint8))
    writer.release()
    return path


def test_video_info_and_frames(tiny_video):
    info = video_info(tiny_video)
    assert (info.width, info.height, info.num_frames) == (32, 24, 5)
    assert [i for i, _ in iter_frames(tiny_video)] == [0, 1, 2, 3, 4]
    assert abs(int(read_frame(tiny_video, 3).mean()) - 150) < 10
