"""Package, verify and install the shared The Way Up archive distributed via the team Google Drive.

The archive ``wayup.zip`` holds the processed subset of the original dataset
(every file except the ``*_large.mp4`` 4K videos) under ``wayup/``.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

from holdon.data.wayup import ROUTES

ARCHIVE_ROOT: str = "wayup"
PARTICIPANT_FORM: str = "participant_form_data.csv"
TOP_LEVEL_FILES: tuple[str, ...] = (
    "README.md", PARTICIPANT_FORM, "topo_green.png", "topo_orange.png",
)
PER_ROUTE_SUFFIXES: tuple[str, ...] = (
    ".mp4", "_holdUsage.csv", "_routeAnnotation.json", "_bolt_holes.json",
)
RECORDINGS: tuple[str, ...] = (
    "p1", "p2a", "p2b", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
)


def expected_files() -> list[str]:
    """Relative paths of every file the shared subset contains."""
    files = list(TOP_LEVEL_FILES)
    for rec in RECORDINGS:
        for route in ROUTES:
            files.extend(f"{rec}/{route}{suffix}" for suffix in PER_ROUTE_SUFFIXES)
    return sorted(files)


def check_layout(root: str | Path) -> list[str]:
    """Return a list of problems with a local dataset folder; empty means it is complete."""
    root = Path(root)
    if not root.is_dir():
        return [f"{root} does not exist"]
    problems = []
    for rel in expected_files():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing {rel}")
        elif path.stat().st_size == 0:
            problems.append(f"empty {rel}")
    problems.extend(
        f"unexpected 4K video {p.relative_to(root)}" for p in root.glob("p*/*_large.mp4")
    )
    return problems


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_subset(root: str | Path, zip_path: str | Path) -> str:
    """Zip the shared subset of ``root`` under ``wayup/`` and return the archive's SHA-256.

    Members are stored without compression (videos are already compressed), in sorted
    order and with a fixed timestamp, so the same input always gives the same archive.
    """
    root, zip_path = Path(root), Path(zip_path)
    problems = check_layout(root)
    if problems:
        raise ValueError(f"Dataset at {root} is incomplete: {problems[:5]}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for rel in expected_files():
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{rel}", date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            with (root / rel).open("rb") as src, zf.open(info, "w") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
    return sha256_file(zip_path)


def install_archive(zip_path: str | Path, dest: str | Path, expected_sha256: str) -> Path:
    """Verify ``zip_path`` against ``expected_sha256``, extract it and check the layout.

    The archive's ``wayup/`` folder is extracted to ``dest`` (e.g. ``data/raw/wayup``).
    Raises ValueError on a checksum mismatch, an unexpected member or an incomplete layout.
    """
    zip_path, dest = Path(zip_path), Path(dest)
    actual = sha256_file(zip_path)
    if actual != expected_sha256.lower():
        raise ValueError(
            f"Checksum mismatch for {zip_path}: got {actual}, expected {expected_sha256}. "
            "The download is corrupt or a different version; download it again."
        )
    prefix = f"{ARCHIVE_ROOT}/"
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = info.filename[len(prefix):] if info.filename.startswith(prefix) else None
            if not rel or ".." in Path(rel).parts:
                raise ValueError(f"Unexpected archive member {info.filename!r}")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
    problems = check_layout(dest)
    if problems:
        raise ValueError(f"Installed dataset at {dest} is incomplete: {problems[:5]}")
    return dest
