"""Integrity checks for the docs OSS thank-you strip (docs/assets/credits.json).

Keeps the curated credits list honest without hand-syncing it to any other file:

* every entry has the required fields,
* ``cargo`` entries name a crate that actually appears in ``Cargo.lock``,
* ``bundled`` entries (shipped binaries) are also listed in
  ``THIRD_PARTY_NOTICES.md`` — the legal doc stays the source of truth for what
  we redistribute; this just cross-checks the two agree.

This runs in the gate via ``pytest tools/docs/`` (see the justfile).
"""

from __future__ import annotations

import json
import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CREDITS_PATH = ROOT / "docs" / "assets" / "credits.json"

REQUIRED_FIELDS = {"name", "url", "used_for", "license", "kind"}
VALID_KINDS = {"cargo", "bundled", "tool"}


def _credits() -> list[dict]:
    return json.loads(CREDITS_PATH.read_text(encoding="utf-8"))


def _cargo_lock_crate_names() -> set[str]:
    data = tomllib.loads((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    return {pkg["name"].lower() for pkg in data.get("package", [])}


def _third_party_notices() -> str:
    return (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")


def test_credits_is_a_nonempty_list() -> None:
    items = _credits()
    assert isinstance(items, list) and items, "credits.json must be a non-empty list"


@pytest.mark.parametrize("entry", _credits(), ids=lambda e: e.get("name", "?"))
def test_entry_schema(entry: dict) -> None:
    missing = REQUIRED_FIELDS - entry.keys()
    assert not missing, f"{entry.get('name')}: missing fields {missing}"
    for field in REQUIRED_FIELDS:
        assert isinstance(entry[field], str) and entry[field].strip(), (
            f"{entry.get('name')}: field {field!r} must be a non-empty string"
        )
    assert entry["url"].startswith("https://"), f"{entry['name']}: url must be https"
    assert entry["kind"] in VALID_KINDS, f"{entry['name']}: bad kind {entry['kind']!r}"
    if entry["kind"] == "cargo":
        assert entry.get("crate"), f"{entry['name']}: cargo entry needs a 'crate' field"


def test_no_duplicate_names() -> None:
    names = [e["name"] for e in _credits()]
    assert len(names) == len(set(names)), "duplicate names in credits.json"


def test_cargo_entries_are_real_dependencies() -> None:
    crates = _cargo_lock_crate_names()
    unknown = [
        e["crate"]
        for e in _credits()
        if e["kind"] == "cargo" and e["crate"].lower() not in crates
    ]
    assert not unknown, f"credits.json lists crates not in Cargo.lock: {unknown}"


def test_bundled_entries_are_in_third_party_notices() -> None:
    notices = _third_party_notices()
    missing = [
        e["name"]
        for e in _credits()
        if e["kind"] == "bundled" and e["name"] not in notices
    ]
    assert not missing, (
        f"bundled entries absent from THIRD_PARTY_NOTICES.md: {missing}"
    )
