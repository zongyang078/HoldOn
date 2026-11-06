"""Tests for YAML config loading."""

from __future__ import annotations

from pathlib import Path

from holdon.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_all_repo_configs_load():
    for path in CONFIG_DIR.glob("*.yaml"):
        assert isinstance(load_config(path), dict)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_config(path) == {}
