"""Tests for the shared contact-event format."""

from __future__ import annotations

import pytest

from holdon.events import ContactEvent, read_events, write_events


def test_round_trip(tmp_path, sample_events):
    path = tmp_path / "events.csv"
    write_events(sample_events, path)
    assert read_events(path) == sample_events


def test_num_frames_is_inclusive():
    assert ContactEvent("v", "left_hand", "h", 5, 5).num_frames == 1


@pytest.mark.parametrize("start,end", [(-1, 3), (5, 4)])
def test_invalid_interval(start, end):
    with pytest.raises(ValueError):
        ContactEvent("v", "left_hand", "h", start, end)


def test_unknown_limb():
    with pytest.raises(ValueError):
        ContactEvent("v", "head", "h", 0, 1)
