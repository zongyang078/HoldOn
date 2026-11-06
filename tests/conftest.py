"""Shared pytest fixtures: tiny synthetic data only, CPU only, no real dataset."""

from __future__ import annotations

import pytest

from holdon.events import ContactEvent


@pytest.fixture
def sample_events() -> list[ContactEvent]:
    """A handful of hand-written events for one synthetic video."""
    return [
        ContactEvent("p0_green", "left_hand", "h1", 10, 30, 0.9),
        ContactEvent("p0_green", "right_foot", "h7", 0, 50, 0.6),
    ]
