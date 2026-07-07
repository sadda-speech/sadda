"""Python-surface tests for sadda.align.mfa (A2 — MFA passthrough).

The MFA subprocess itself is gated (mfa is a heavy conda/Kaldi install absent in
CI), so the real align path skips. The unit-testable seams — TextGrid → Alignment
mapping, the not-installed error, and import_alignment into a bundle — run
everywhere via a canned MFA-style TextGrid.
"""

from __future__ import annotations

import shutil
import tempfile
import wave
from pathlib import Path

import pytest

import sadda
from sadda import align

pytestmark = pytest.mark.filterwarnings("ignore::sadda.ProvisionalAPIWarning")


# A minimal long-format TextGrid shaped like MFA output: a "words" tier with an
# empty leading/trailing gap around one word, and a "phones" tier whose silence
# is *labelled* (sil/sp) — the modeled-silence case.
_MFA_TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.2
            text = ""
        intervals [2]:
            xmin = 0.2
            xmax = 0.6
            text = "hi"
        intervals [3]:
            xmin = 0.6
            xmax = 1.0
            text = ""
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 4
        intervals [1]:
            xmin = 0
            xmax = 0.2
            text = "sil"
        intervals [2]:
            xmin = 0.2
            xmax = 0.4
            text = "h"
        intervals [3]:
            xmin = 0.4
            xmax = 0.6
            text = "aɪ"
        intervals [4]:
            xmin = 0.6
            xmax = 1.0
            text = "sp"
'''


def _write_textgrid(td: Path) -> Path:
    p = td / "hi.TextGrid"
    p.write_text(_MFA_TEXTGRID, encoding="utf-8")
    return p


def _write_wav(path: Path, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * sample_rate)


# --- TextGrid → Alignment (the parse seam; no MFA needed) ---


def test_alignment_from_textgrid_maps_words_and_phones() -> None:
    with tempfile.TemporaryDirectory() as td:
        tg = _write_textgrid(Path(td))
        al = align.alignment_from_textgrid(tg)

    # phones are the full contiguous partition, labels verbatim
    assert [p.label for p in al.phones] == ["sil", "h", "aɪ", "sp"]
    assert al.phones[0].start_seconds == 0.0
    assert al.phones[-1].end_seconds == pytest.approx(1.0)
    # words: empty gaps stay empty; the real word wraps only its own phones
    assert [w.text for w in al.words] == ["", "hi", ""]
    hi = al.words[1]
    assert [p.label for p in hi.phones] == ["h", "aɪ"]
    # MFA gives no per-phone score
    assert all(p.score is None for p in al.phones)


def test_modeled_silence_keeps_its_label() -> None:
    # The whole point of A2's silence: modeled sil/sp are labelled, NOT blanked
    # (contrast the neural aligner's imputed empty intervals).
    with tempfile.TemporaryDirectory() as td:
        al = align.alignment_from_textgrid(_write_textgrid(Path(td)))
    silences = [p for p in al.phones if p.label in ("sil", "sp")]
    assert len(silences) == 2
    assert all(p.label != "" for p in silences)


def test_alignment_from_textgrid_missing_tier_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        tg = _write_textgrid(Path(td))
        with pytest.raises(ValueError, match="phone tier"):
            align.alignment_from_textgrid(tg, phone_tier="segments")


# --- import_alignment into a bundle (backend-agnostic; no MFA needed) ---


def test_import_alignment_writes_word_and_phone_tiers() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        al = align.alignment_from_textgrid(_write_textgrid(tdp))
        proj = sadda.new_project(tdp / "p", "demo")
        wav = tdp / "hi.wav"
        _write_wav(wav)
        bundle_id = proj.add_bundle("hi", wav)

        wt, pt = align.import_alignment(proj, bundle_id, al)

        words = proj.intervals(wt)
        phones = proj.intervals(pt)
        # word tier: empty gaps are unlabeled (None), the word carries its text
        assert [w.label for w in words] == [None, "hi", None]
        # phone tier: modeled silence keeps its label through the round-trip
        assert [p.label for p in phones] == ["sil", "h", "aɪ", "sp"]
        # spans preserved
        assert words[1].start_seconds == pytest.approx(0.2)
        assert phones[0].start_seconds == 0.0


def test_import_alignment_flat_when_not_hierarchical() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        al = align.alignment_from_textgrid(_write_textgrid(tdp))
        proj = sadda.new_project(tdp / "p", "demo")
        wav = tdp / "hi.wav"
        _write_wav(wav)
        bundle_id = proj.add_bundle("hi", wav)

        wt, pt = align.import_alignment(proj, bundle_id, al, hierarchical=False)
        assert len(proj.intervals(wt)) == 3
        assert len(proj.intervals(pt)) == 4


# --- not-installed error (no MFA needed) ---


def test_mfa_align_errors_clearly_when_mfa_absent(monkeypatch) -> None:
    import sadda.align.mfa as mfa

    monkeypatch.setattr(mfa.shutil, "which", lambda _name: None)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        _write_wav(wav)
        with pytest.raises(FileNotFoundError, match="Montreal Forced Aligner"):
            mfa.mfa_align(wav, "hi")


# --- gated integration (real MFA + downloaded models) ---


_HAVE_MFA = shutil.which("mfa") is not None


@pytest.mark.skipif(not _HAVE_MFA, reason="mfa binary not installed")
def test_mfa_align_smoke_requires_models() -> None:
    # Only runs where MFA + the english_mfa models are installed; a light check
    # that the single-file path returns an Alignment with words and phones.
    import os

    audio = os.environ.get("SADDA_TEST_MFA_AUDIO")
    transcript = os.environ.get("SADDA_TEST_MFA_TRANSCRIPT")
    if not (audio and transcript):
        pytest.skip("SADDA_TEST_MFA_AUDIO/TRANSCRIPT not set")
    al = align.mfa_align(audio, transcript)
    assert al.words and al.phones
