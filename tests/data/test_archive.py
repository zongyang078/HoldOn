"""Tests for packaging, verifying and installing the shared dataset archive (tiny fake data)."""

from __future__ import annotations

import zipfile

import pytest

from holdon.data import archive


@pytest.fixture
def fake_dataset(tmp_path):
    root = tmp_path / "source"
    for rel in archive.expected_files():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rel.encode())
    return root


def test_expected_files_count():
    files = archive.expected_files()
    assert len(files) == 92 and archive.PARTICIPANT_FORM in files


def test_check_layout(fake_dataset):
    assert archive.check_layout(fake_dataset) == []
    (fake_dataset / archive.PARTICIPANT_FORM).unlink()
    assert archive.check_layout(fake_dataset) == [f"missing {archive.PARTICIPANT_FORM}"]
    (fake_dataset / archive.PARTICIPANT_FORM).write_text("x")
    (fake_dataset / "p1" / "green_large.mp4").write_bytes(b"big")
    (fake_dataset / "p3" / "orange_holdUsage.csv").unlink()
    (fake_dataset / "p4" / "green.mp4").write_bytes(b"")
    assert archive.check_layout(fake_dataset) == [
        "missing p3/orange_holdUsage.csv",
        "empty p4/green.mp4",
        "unexpected 4K video p1/green_large.mp4",
    ]
    assert archive.check_layout(fake_dataset / "nope") != []


def test_package_is_complete_and_deterministic(tmp_path, fake_dataset):
    sha_a = archive.package_subset(fake_dataset, tmp_path / "a.zip")
    sha_b = archive.package_subset(fake_dataset, tmp_path / "b.zip")
    assert sha_a == sha_b
    names = zipfile.ZipFile(tmp_path / "a.zip").namelist()
    assert len(names) == 92 and all(n.startswith("wayup/") for n in names)
    assert "wayup/participant_form_data.csv" in names


def test_install_round_trip(tmp_path, fake_dataset):
    zip_path = tmp_path / "wayup.zip"
    sha = archive.package_subset(fake_dataset, zip_path)
    dest = archive.install_archive(zip_path, tmp_path / "data" / "raw" / "wayup", sha)
    assert archive.check_layout(dest) == []
    assert (dest / "p2b" / "orange.mp4").read_bytes() == b"p2b/orange.mp4"


def test_install_rejects_bad_checksum(tmp_path, fake_dataset):
    zip_path = tmp_path / "wayup.zip"
    archive.package_subset(fake_dataset, zip_path)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        archive.install_archive(zip_path, tmp_path / "out", "0" * 64)
    assert not (tmp_path / "out").exists()


def test_install_rejects_path_traversal(tmp_path):
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("wayup/../../escape.txt", "x")
    with pytest.raises(ValueError, match="Unexpected archive member"):
        archive.install_archive(zip_path, tmp_path / "out", archive.sha256_file(zip_path))
    assert not (tmp_path / "escape.txt").exists()
