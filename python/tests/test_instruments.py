"""A3 — instrument calibration + calibrated SPL, Python surface."""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path

import sadda


def _write_short_wav(path: Path, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * sample_rate, *([0] * sample_rate)))


def test_calibration_math() -> None:
    cal = sadda.Calibration(reference_spl_db=94.0, reference_db_fs=-20.0)
    assert abs(cal.spl_offset_db() - 114.0) < 1e-9
    # -26 dB-FS + 114 dB offset → 88 dB-SPL.
    assert abs(cal.to_spl(-26.0) - 88.0) < 1e-4


def test_instrument_roundtrip_and_bundle_resolution() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = sadda.new_project(Path(td) / "p", "demo")
        cal = sadda.Calibration(reference_spl_db=94.0, reference_db_fs=-20.0)
        iid = proj.add_instrument("B&K 4189", kind="microphone", calibration=cal)

        got = proj.get_instrument(iid)
        assert got.name == "B&K 4189"
        assert got.calibration is not None
        assert abs(got.calibration.reference_spl_db - 94.0) < 1e-9
        assert len(proj.instruments()) == 1

        # bundle → session → instrument resolves the calibration.
        sid = proj.add_session("s1", instrument_id=iid)
        wav = Path(td) / "tone.wav"
        _write_short_wav(wav)
        bid = proj.add_bundle("greeting", wav, session_id=sid)
        resolved = proj.bundle_calibration(bid)
        assert resolved is not None
        assert abs(resolved.spl_offset_db() - 114.0) < 1e-9

        # A bundle with no session is uncalibrated.
        wav2 = Path(td) / "tone2.wav"
        _write_short_wav(wav2)
        uncal = proj.add_bundle("plain", wav2)
        assert proj.bundle_calibration(uncal) is None
