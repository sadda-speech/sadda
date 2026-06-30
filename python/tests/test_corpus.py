"""Python-surface tests for the B1 corpus API (Project / Speaker / Session /
Bundle FKs / audit user)."""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path

import pytest
import sadda


def _write_short_wav(path: Path, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = bytes()
        for i in range(sample_rate // 4):
            sample = int(0.3 * 32767 * ((i % 100) / 100.0 - 0.5))
            frames += struct.pack("<h", sample)
        w.writeframes(frames)


def test_new_project_smoke_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = sadda.new_project(Path(td) / "p", "demo")
        assert proj.name == "demo"
        assert proj.root.endswith("p")
        assert sadda.SCHEMA_VERSION >= 3


def test_open_project_round_trip_via_disk() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "p"
        sadda.new_project(root, "demo")
        reopened = sadda.open_project(root)
        assert reopened.name == "demo"


def test_add_speaker_and_list() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = sadda.new_project(Path(td) / "p", "demo")
        alice = proj.add_speaker("Alice", sex="f", birth_year=1990)
        bob = proj.add_speaker("Bob")
        speakers = proj.speakers()
        assert [s.id for s in speakers] == [alice, bob]
        assert speakers[0].name == "Alice"
        assert speakers[0].sex == "f"
        assert speakers[0].birth_year == 1990
        assert speakers[1].sex is None
        assert proj.get_speaker(alice).name == "Alice"


def test_add_session_and_list() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = sadda.new_project(Path(td) / "p", "demo")
        sid = proj.add_session(
            "lab1",
            location="room_b",
            started_at="2026-05-21T14:00:00Z",
            notes="initial check",
        )
        sessions = proj.sessions()
        assert len(sessions) == 1
        assert sessions[0].id == sid
        assert sessions[0].location == "room_b"
        assert sessions[0].started_at == "2026-05-21T14:00:00Z"
        assert proj.get_session(sid).name == "lab1"


def test_add_bundle_with_session_and_speaker() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = sadda.new_project(Path(td) / "p", "demo")
        speaker_id = proj.add_speaker("Alice")
        session_id = proj.add_session("lab1")
        wav = Path(td) / "tone.wav"
        _write_short_wav(wav)
        bundle_id = proj.add_bundle(
            "greeting",
            wav,
            session_id=session_id,
            speaker_id=speaker_id,
            extra='{"take":2}',
        )
        bundles = proj.bundles()
        assert len(bundles) == 1
        assert bundles[0].id == bundle_id
        assert bundles[0].session_id == session_id
        assert bundles[0].speaker_id == speaker_id
        assert bundles[0].extra == '{"take":2}'


def test_delete_bundle_cascades_and_removes_wav() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = sadda.new_project(Path(td) / "p", "demo")
        wav = Path(td) / "tone.wav"
        _write_short_wav(wav)
        bundle_id = proj.add_bundle("greeting", wav)
        assert len(proj.bundles()) == 1
        audio_rel = proj.bundles()[0].audio_relative_path
        audio_abs = Path(proj.root) / audio_rel
        assert audio_abs.exists()
        proj.delete_bundle(bundle_id)
        assert proj.bundles() == []
        assert not audio_abs.exists()


def test_delete_bundle_is_idempotent_on_missing_id() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = sadda.new_project(Path(td) / "p", "demo")
        # No bundles exist; deleting a non-existent id must not raise.
        proj.delete_bundle(9_999)


def test_rename_bundle_updates_name_and_keeps_wav() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = sadda.new_project(Path(td) / "p", "demo")
        wav = Path(td) / "tone.wav"
        _write_short_wav(wav)
        bundle_id = proj.add_bundle("greeting", wav)
        audio_rel = proj.bundles()[0].audio_relative_path

        proj.rename_bundle(bundle_id, "  farewell  ")
        assert proj.bundles()[0].name == "farewell"
        # The WAV path is keyed independently of the display name.
        assert proj.bundles()[0].audio_relative_path == audio_rel

        # Empty name and unknown id both raise.
        with pytest.raises(Exception):
            proj.rename_bundle(bundle_id, "   ")
        with pytest.raises(Exception):
            proj.rename_bundle(9_999, "x")


def test_audit_user_default_and_setter() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = sadda.new_project(Path(td) / "p", "demo")
        assert proj.audit_user == "local"
        proj.set_audit_user("alice")
        assert proj.audit_user == "alice"


def test_corpus_surface_is_stable() -> None:
    from sadda._stability import get_stability

    for sym in (
        sadda.new_project,
        sadda.open_project,
        sadda.Project,
        sadda.Bundle,
        sadda.Speaker,
        sadda.Session,
    ):
        assert get_stability(sym) == "stable", sym
    # SCHEMA_VERSION is a plain value, not a tiered callable.
    assert isinstance(sadda.SCHEMA_VERSION, int)
