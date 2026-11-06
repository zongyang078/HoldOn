"""Shared contact-event format used by every track and by the evaluation module.

The format is specified in docs/event_format.md. Changing it requires team agreement.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

LIMBS: tuple[str, ...] = ("left_hand", "right_hand", "left_foot", "right_foot")


@dataclass(frozen=True)
class ContactEvent:
    """One contact of a limb with a hold over an inclusive frame interval."""

    video_id: str
    limb: str
    hold_id: str
    start_frame: int
    end_frame: int
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.limb not in LIMBS:
            raise ValueError(f"Unknown limb {self.limb!r}; expected one of {LIMBS}")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError(
                f"Invalid interval [{self.start_frame}, {self.end_frame}]"
            )

    @property
    def num_frames(self) -> int:
        """Length of the inclusive interval in frames."""
        return self.end_frame - self.start_frame + 1


FIELDNAMES: tuple[str, ...] = tuple(f.name for f in fields(ContactEvent))


def write_events(events: Iterable[ContactEvent], path: str | Path) -> None:
    """Write events to a CSV file with the shared column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def read_events(path: str | Path) -> list[ContactEvent]:
    """Read events from a CSV file written by write_events."""
    with Path(path).open(newline="") as fh:
        return [
            ContactEvent(
                video_id=row["video_id"],
                limb=row["limb"],
                hold_id=row["hold_id"],
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                score=float(row["score"]),
            )
            for row in csv.DictReader(fh)
        ]
