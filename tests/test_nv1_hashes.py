
import hashlib
from pathlib import Path

import pytest

from scripts import check_dataset


def test_sha256_matches_actual_file(tmp_path):
    image = tmp_path / "test.jpg"
    content = b"example image bytes"
    image.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    actual = check_dataset.compute_sha256(image)

    assert actual == expected


def test_sha256_detects_modified_file(tmp_path):
    image = tmp_path / "test.jpg"
    image.write_bytes(b"original content")

    original_hash = check_dataset.compute_sha256(image)

    image.write_bytes(b"modified content")

    assert check_dataset.compute_sha256(image) != original_hash


def test_verify_hashes_requires_data_root():
    with pytest.raises(
        ValueError,
        match="requires --data-root"
    ):
        check_dataset.check(verify_hashes=True)


def test_verify_hashes_rejects_mismatch(tmp_path, monkeypatch):
    first_row = check_dataset.load_csv(
        check_dataset.ROOT / "data/dataset_manifest.csv"
    )[0]

    expected = first_row["sha256"].strip().lower()
    wrong_hash = "0" * 64
    if expected == wrong_hash:
        wrong_hash = "1" * 64

    # Simulate image existence without downloading 6502 images.
    monkeypatch.setattr(Path, "is_file", lambda self: True)

    # Simulate a mismatched image hash.
    monkeypatch.setattr(
        check_dataset,
        "compute_sha256",
        lambda path: wrong_hash,
    )

    with pytest.raises(
        AssertionError,
        match="SHA-256 mismatch"
    ):
        check_dataset.check(
            data_root=tmp_path,
            verify_hashes=True,
        )
