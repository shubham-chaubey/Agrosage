"""Unit tests for password hashing in auth.py (pure, no DB / no network)."""
import hashlib

from auth import _hash


def test_hash_is_deterministic():
    assert _hash("password123") == _hash("password123")


def test_hash_has_sha256_length():
    assert len(_hash("anything")) == 64


def test_hash_differs_for_different_input():
    assert _hash("password123") != _hash("password124")


def test_hash_matches_sha256():
    expected = hashlib.sha256("agrosage".encode("utf-8")).hexdigest()
    assert _hash("agrosage") == expected


def test_hash_handles_unicode():
    digest = _hash("किसान")
    assert isinstance(digest, str) and len(digest) == 64
