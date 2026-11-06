"""Leave-one-participant-out (LOPO) cross-validation splits. Never split by random frames.

A participant can have several recordings (``p2a`` and ``p2b`` are both participant 2);
all of them always fall into the same fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class Fold:
    """One LOPO fold: the held-out participant and the recordings on each side."""

    test_participant: int
    train: tuple[str, ...]
    test: tuple[str, ...]


def lopo_folds(recordings: Iterable[str], participant_of: Callable[[str], int]) -> list[Fold]:
    """Build one fold per participant, holding out all of that participant's recordings."""
    recordings = sorted(set(recordings))
    participants = sorted({participant_of(r) for r in recordings})
    folds = []
    for p in participants:
        test = tuple(r for r in recordings if participant_of(r) == p)
        train = tuple(r for r in recordings if participant_of(r) != p)
        folds.append(Fold(p, train, test))
    return folds
