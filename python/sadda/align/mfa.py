"""sadda.align.mfa — gold-standard forced alignment via the Montreal Forced Aligner.

A passthrough to an external **MFA 3.x** install (Kaldi/HMM-GMM under the hood):
sadda shells out to the ``mfa`` CLI, then imports the resulting TextGrid into the
same :class:`~sadda.align.aligner.Alignment` the neural aligner returns
(:func:`sadda.align.align`), so the two backends are interchangeable. MFA is a
heavy conda/Kaldi tool, so it is strictly **opt-in** and never a sadda dependency —
if ``mfa`` isn't on ``PATH`` the functions raise an actionable error (the same
detected-or-clear-error pattern as espeak-ng and the ``sadda[align]`` model fetch).

Two entry points:

- :func:`mfa_align` — one ``(audio, transcript)`` pair via ``mfa align_one``
  (MFA 3.0's fast single-file command, no corpus/database overhead).
- :func:`mfa_align_corpus` — a directory of paired audio + transcript files via
  ``mfa align``, where MFA's batch optimisations (and speaker adaptation) apply.

Both need a pronunciation **dictionary** and a trained **acoustic model**; pass
MFA model *names* (e.g. ``"english_mfa"``, downloaded via ``mfa model download``)
or paths. The IPA-flavoured ``english_mfa`` models are the default so both
backends speak IPA — though MFA's phone inventory is not identical to espeak-ng's,
so labels won't match the neural aligner's byte-for-byte.

**Modeled vs imputed silence.** Unlike the neural aligner, whose blank/VAD
detectors *infer* silence and emit empty intervals, MFA *models* optional silence
in its HMM — so its silence intervals carry real labels (``sil``/``sp``/``spn``).
Those labels are preserved verbatim: a modeled-silence assertion is not the same
claim as an imputed gap, and shouldn't be blanked.

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union

from sadda import _native
from sadda._stability import provisional

from .aligner import Alignment, TimedPhone, TimedWord

__all__ = ["mfa_align", "mfa_align_corpus", "alignment_from_textgrid"]

#: Default MFA dictionary + acoustic model (the IPA-flavoured English models).
DEFAULT_DICTIONARY = "english_mfa"
DEFAULT_ACOUSTIC_MODEL = "english_mfa"

_PathLike = Union[str, Path]


def _mfa_binary() -> str:
    """Resolve the ``mfa`` executable or raise with an install hint."""
    binary = shutil.which("mfa")
    if binary is None:
        raise FileNotFoundError(
            "Montreal Forced Aligner ('mfa') not found on PATH. MFA is an "
            "optional gold-standard backend — install it (conda recommended: "
            "`conda install -c conda-forge montreal-forced-aligner`), then fetch "
            "models with `mfa model download acoustic english_mfa` and "
            "`mfa model download dictionary english_mfa`."
        )
    return binary


def _run_mfa(args: Sequence[str]) -> None:
    """Run ``mfa <args>``, raising a readable error on non-zero exit."""
    proc = subprocess.run(
        [_mfa_binary(), *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        detail = "\n".join(tail)
        hint = ""
        if "not found" in detail.lower() or "does not exist" in detail.lower():
            hint = (
                "\nIf a dictionary or acoustic model is missing, download it with "
                "`mfa model download`."
            )
        raise RuntimeError(f"mfa {args[0]} failed (exit {proc.returncode}):\n{detail}{hint}")


# [docs:sadda.align.mfa.alignment_from_textgrid]
def alignment_from_textgrid(
    path: _PathLike,
    *,
    word_tier: str = "words",
    phone_tier: str = "phones",
) -> Alignment:
    """Build an :class:`Alignment` from an aligner's TextGrid output.

    Reads the ``word_tier`` and ``phone_tier`` interval tiers (matched by name,
    case-insensitively) via the engine's TextGrid parser, and wraps each word's
    contained phones (by interval midpoint — the tiers partition the recording
    contiguously). Labels are preserved verbatim, so MFA's modeled-silence marks
    (``sil``/``sp``) stay labelled while word-level gaps stay empty. Kept as a
    standalone function so this mapping is unit-testable without running MFA.
    """
    tiers = {name.lower(): rows for name, rows in _native.parse_textgrid_intervals(str(path))}
    if word_tier.lower() not in tiers:
        raise ValueError(
            f"word tier {word_tier!r} not in TextGrid (tiers: {sorted(tiers)})"
        )
    if phone_tier.lower() not in tiers:
        raise ValueError(
            f"phone tier {phone_tier!r} not in TextGrid (tiers: {sorted(tiers)})"
        )

    phones = tuple(
        TimedPhone(label=label, start_seconds=start, end_seconds=end)
        for (start, end, label) in tiers[phone_tier.lower()]
    )

    def _phones_within(w_start: float, w_end: float) -> tuple[TimedPhone, ...]:
        return tuple(
            p
            for p in phones
            if w_start <= (p.start_seconds + p.end_seconds) / 2.0 < w_end
        )

    words = tuple(
        TimedWord(
            text=label,
            start_seconds=start,
            end_seconds=end,
            phones=_phones_within(start, end),
        )
        for (start, end, label) in tiers[word_tier.lower()]
    )
    return Alignment(words=words, phones=phones)


# [docs:sadda.align.mfa.mfa_align]
@provisional
def mfa_align(
    audio_path: _PathLike,
    transcript: str,
    *,
    dictionary: _PathLike = DEFAULT_DICTIONARY,
    acoustic_model: _PathLike = DEFAULT_ACOUSTIC_MODEL,
    g2p_model: Optional[_PathLike] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Alignment:
    """Force-align one recording to ``transcript`` with MFA (``mfa align_one``).

    ``audio_path`` is a WAV file; ``transcript`` is its orthographic text.
    ``dictionary`` and ``acoustic_model`` are MFA model names (default the IPA
    ``english_mfa`` pair) or paths; ``g2p_model`` supplies pronunciations for
    out-of-vocabulary words. Returns an :class:`Alignment`; write it onto a
    bundle with :func:`sadda.align.import_alignment`.

    Note: ``align_one`` skips speaker adaptation, so it is marginally less precise
    than full-corpus :func:`mfa_align_corpus` — the price of single-file speed.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    with tempfile.TemporaryDirectory(prefix="sadda_mfa_") as tmp:
        tmp_dir = Path(tmp)
        text_file = tmp_dir / f"{audio_path.stem}.txt"
        text_file.write_text(transcript, encoding="utf-8")
        out_tg = tmp_dir / f"{audio_path.stem}.TextGrid"

        args = ["align_one"]
        if g2p_model is not None:
            args += ["--g2p_model_path", str(g2p_model)]
        if extra_args:
            args += list(extra_args)
        args += [
            str(audio_path),
            str(text_file),
            str(dictionary),
            str(acoustic_model),
            str(out_tg),
        ]
        _run_mfa(args)

        if not out_tg.exists():
            raise RuntimeError(f"mfa align_one produced no output at {out_tg}")
        return alignment_from_textgrid(out_tg)


# [docs:sadda.align.mfa.mfa_align_corpus]
@provisional
def mfa_align_corpus(
    corpus_dir: _PathLike,
    *,
    dictionary: _PathLike = DEFAULT_DICTIONARY,
    acoustic_model: _PathLike = DEFAULT_ACOUSTIC_MODEL,
    output_dir: Optional[_PathLike] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> dict[str, Alignment]:
    """Batch-align an MFA corpus directory with ``mfa align``.

    ``corpus_dir`` follows MFA's layout — paired audio and transcript files
    (``utt.wav`` + ``utt.lab``/``utt.txt``), optionally in per-speaker
    subdirectories. Runs the full pipeline (speaker adaptation included), then
    parses every output TextGrid. Returns ``{file_stem: Alignment}``. If
    ``output_dir`` is given the TextGrids are kept there; otherwise a temp dir is
    used and discarded after parsing.

    Prefer this over per-file :func:`mfa_align` for many recordings: MFA amortises
    model loading and applies corpus-level speaker adaptation.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise NotADirectoryError(f"corpus directory not found: {corpus_dir}")

    def _run_and_collect(out: Path) -> dict[str, Alignment]:
        args = ["align"]
        if extra_args:
            args += list(extra_args)
        args += [str(corpus_dir), str(dictionary), str(acoustic_model), str(out)]
        _run_mfa(args)
        results: dict[str, Alignment] = {}
        for tg in sorted(out.rglob("*.TextGrid")):
            results[tg.stem] = alignment_from_textgrid(tg)
        if not results:
            raise RuntimeError(f"mfa align produced no TextGrids under {out}")
        return results

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return _run_and_collect(out)
    with tempfile.TemporaryDirectory(prefix="sadda_mfa_corpus_") as tmp:
        return _run_and_collect(Path(tmp))
