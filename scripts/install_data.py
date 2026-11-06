"""Install the shared The Way Up archive downloaded from the team Google Drive.

Verifies the SHA-256 checksum from configs/paths.yaml, extracts the archive to the
``raw.wayup`` folder and checks that all expected files are present.

Usage:
    python3 scripts/install_data.py ~/Downloads/wayup.zip [--paths configs/paths.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from holdon.config import load_config
from holdon.data.archive import install_archive


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", help="path to the downloaded wayup.zip")
    parser.add_argument("--paths", default="configs/paths.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    paths = load_config(args.paths)
    dest = Path(paths["raw"]["wayup"])
    try:
        install_archive(args.archive, dest, paths["shared"]["wayup_zip_sha256"])
    except ValueError as err:
        logging.error("%s", err)
        sys.exit(1)
    logging.info("Dataset installed and complete at %s", dest)


if __name__ == "__main__":
    main()
