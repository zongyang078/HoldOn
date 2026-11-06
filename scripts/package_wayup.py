"""Package the local The Way Up subset into one zip for the team Google Drive.

The archive contains every file of the original dataset except *_large.mp4.
Writes ``<zip>.sha256`` next to the archive; put the digest into configs/paths.yaml
(``shared.wayup_zip_sha256``) and upload the zip to the team Drive.

Usage:
    python3 scripts/package_wayup.py [--paths configs/paths.yaml] [--out data/share/wayup.zip]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from holdon.config import load_config
from holdon.data.archive import package_subset


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--out", default="data/share/wayup.zip")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = Path(load_config(args.paths)["raw"]["wayup"])
    out = Path(args.out)
    digest = package_subset(root, out)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    logging.info("Wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    logging.info("SHA-256 %s", digest)


if __name__ == "__main__":
    main()
