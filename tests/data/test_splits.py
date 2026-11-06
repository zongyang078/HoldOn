"""Tests for leave-one-participant-out splits."""

from __future__ import annotations

from holdon.data.splits import lopo_folds
from holdon.data.wayup import participant_of


def test_lopo_keeps_participant_recordings_together():
    folds = lopo_folds(["p1", "p2a", "p2b", "p3"], participant_of)
    assert [f.test_participant for f in folds] == [1, 2, 3]
    fold2 = folds[1]
    assert fold2.test == ("p2a", "p2b")
    assert set(fold2.train) == {"p1", "p3"}
    for f in folds:
        assert not set(f.train) & set(f.test)
