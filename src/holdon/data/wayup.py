"""Load The Way Up recordings: processed videos, hold-usage labels and route annotations.

The file formats are documented in docs/dataset_schema.md. Only the processed
720x1280 / 25 fps videos are used; ``*_large.mp4`` files are ignored.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from holdon.events import ContactEvent

logger = logging.getLogger(__name__)

ROUTES: tuple[str, ...] = ("green", "orange")
FPS: int = 25

LIMB_CODES: dict[str, str] = {
    "lh": "left_hand",
    "rh": "right_hand",
    "lf": "left_foot",
    "rf": "right_foot",
}
WALL_HOLD_ID: str = "w"
"""Hold id used when the climber pushes on the wall instead of a hold."""

_LABEL_RE = re.compile(r"\s*([lr][hf])\s*(\d+|w)\s*")
_RECORDING_RE = re.compile(r"p(\d+)([a-z]?)")


@dataclass(frozen=True)
class Hold:
    """One hold bounding box in pixel coordinates of the processed video."""

    hold_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    is_foothold: bool


@dataclass(frozen=True)
class LabeledEvent:
    """A ground-truth contact event plus the frame ranges where the limb is occluded."""

    event: ContactEvent
    occluded: tuple[tuple[int, int], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class WallReference:
    """Bolt-hole correspondences and the homography mapping image pixels to wall coordinates."""

    img_coords: np.ndarray
    world_coords: np.ndarray
    homography: np.ndarray


def participant_of(recording: str) -> int:
    """Return the participant number of a recording folder name, e.g. ``p2b`` -> 2."""
    match = _RECORDING_RE.fullmatch(recording)
    if match is None:
        raise ValueError(f"Not a recording folder name: {recording!r}")
    return int(match.group(1))


def video_id(recording: str, route: str) -> str:
    """Build the video identifier used in ContactEvent, e.g. ``p1_green``."""
    return f"{recording}_{route}"


def list_recordings(root: str | Path) -> list[str]:
    """List recording folder names (``p1``, ``p2a``, ...) under the dataset root."""
    root = Path(root)
    names = [p.name for p in root.iterdir() if p.is_dir() and _RECORDING_RE.fullmatch(p.name)]
    return sorted(names, key=lambda n: (participant_of(n), n))


def video_path(root: str | Path, recording: str, route: str) -> Path:
    """Path of the processed (25 fps) video for one recording and route."""
    return Path(root) / recording / f"{route}.mp4"


def parse_label(label: str) -> tuple[str, str] | None:
    """Parse a label such as ``"rh 2"``, ``"lh2"`` or ``"lf w"`` into (limb, hold_id).

    Returns None when the label does not follow the documented pattern.
    """
    match = _LABEL_RE.fullmatch(label)
    if match is None:
        return None
    return LIMB_CODES[match.group(1)], match.group(2)


def parse_occlusion(text: str) -> tuple[tuple[int, int], ...]:
    """Parse an occlusion field such as ``"59-69, 133-195"`` into inclusive frame ranges."""
    ranges = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        start, sep, end = token.partition("-")
        ranges.append((int(start), int(end)) if sep else (int(start), int(start)))
    return tuple(ranges)


def load_hold_usage(
    path: str | Path, vid: str, num_frames: int | None = None
) -> list[LabeledEvent]:
    """Load a ``*_holdUsage.csv`` file as ground-truth events.

    The first header cell holds the recording name rather than a column name, so
    columns are read by position. Rows whose label cannot be parsed are skipped
    with a warning. If ``num_frames`` is given, end frames are clipped to
    ``num_frames - 1`` (a few files end one frame past the video).
    """
    events: list[LabeledEvent] = []
    with Path(path).open(newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # header: <recording>, startFrame, endFrame, framesOccluded
        for line_no, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue
            parsed = parse_label(row[0])
            if parsed is None:
                logger.warning("%s:%d: skipping unparseable label %r", path, line_no, row[0])
                continue
            limb, hold_id = parsed
            start, end = int(row[1]), int(row[2])
            if num_frames is not None and end >= num_frames:
                logger.warning(
                    "%s:%d: end frame %d clipped to %d", path, line_no, end, num_frames - 1
                )
                end = num_frames - 1
            occluded = tuple(
                (max(a, start), min(b, end))
                for a, b in parse_occlusion(row[3] if len(row) > 3 else "")
                if a <= end
            )
            events.append(
                LabeledEvent(ContactEvent(vid, limb, hold_id, start, end, 1.0), occluded)
            )
    return events


def load_holds(path: str | Path) -> list[Hold]:
    """Load a ``*_routeAnnotation.json`` file. The hold id is the index in the list."""
    with Path(path).open() as fh:
        data = json.load(fh)
    return [
        Hold(
            hold_id=str(i),
            x1=int(h["topLeftCorner"]["x"]),
            y1=int(h["topLeftCorner"]["y"]),
            x2=int(h["bottomRightCorner"]["x"]),
            y2=int(h["bottomRightCorner"]["y"]),
            is_foothold=bool(h["isFoothold"]),
        )
        for i, h in enumerate(data["holds"])
    ]


def load_wall_reference(path: str | Path) -> WallReference:
    """Load a ``*_bolt_holes.json`` file (homography maps image pixels to wall coordinates).

    Most files store four correspondences plus ``homography_mat``. One file
    (``p2a/green``) stores 350 correspondences and no matrix; in that case the
    homography is estimated from the correspondences with RANSAC.
    """
    with Path(path).open() as fh:
        data = json.load(fh)
    img = np.asarray(data["img_coords"], dtype=np.float64)
    world = np.asarray(data["world_coords"], dtype=np.float64)
    if "homography_mat" in data:
        homography = np.asarray(data["homography_mat"], dtype=np.float64)
    else:
        logger.info("%s: no homography_mat, estimating from %d points", path, len(img))
        homography, _ = cv2.findHomography(img, world[:, :2], cv2.RANSAC, 5.0)
        if homography is None:
            raise ValueError(f"Cannot estimate homography from {path}")
    return WallReference(img_coords=img, world_coords=world, homography=homography)


@dataclass
class Recording:
    """All annotation files of one recording and route, plus the processed video path."""

    recording: str
    route: str
    video: Path
    holds: list[Hold]
    events: list[LabeledEvent]
    wall: WallReference

    @property
    def video_id(self) -> str:
        """Identifier used in ContactEvent."""
        return video_id(self.recording, self.route)

    @property
    def participant(self) -> int:
        """Participant number (``p2a`` and ``p2b`` are both participant 2)."""
        return participant_of(self.recording)


def load_recording(
    root: str | Path, recording: str, route: str, num_frames: int | None = None
) -> Recording:
    """Load labels, hold boxes and wall reference of one recording and route."""
    folder = Path(root) / recording
    vid = video_id(recording, route)
    return Recording(
        recording=recording,
        route=route,
        video=video_path(root, recording, route),
        holds=load_holds(folder / f"{route}_routeAnnotation.json"),
        events=load_hold_usage(folder / f"{route}_holdUsage.csv", vid, num_frames),
        wall=load_wall_reference(folder / f"{route}_bolt_holes.json"),
    )
