"""sadda.align.aligner — the forced-alignment orchestrator.

Ties the pieces together (see the 2026-07-05 DEVLOG design entry): espeak-ng G2P
(the target phones) + an :class:`~sadda.align.model.AcousticModel` (per-frame CTC
posteriors) + the engine's forced-align DP (`sadda._native.forced_align`) →
time-aligned **Word** and **Phone** results.

The phone→class-id step is a greedy longest-match of each word's IPA against the
model's vocabulary (so multi-character vocab tokens like ``dʒ`` are matched
whole), which is why G2P stays model-agnostic and the tokenization lives here.
:func:`align` returns a plain :class:`Alignment`; :func:`syllabify` derives a
Syllable tier from it by rule, and :func:`import_alignment` writes one onto a
project's bundle as Word + Phone interval tiers (backend-agnostic — neural or MFA).

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from sadda import _native
from sadda._stability import provisional

from .g2p import phonemize
from .model import AcousticModel

__all__ = [
    "TimedPhone",
    "TimedWord",
    "TimedSyllable",
    "Alignment",
    "tokenize",
    "align",
    "align_auto",
    "syllabify",
    "import_alignment",
]


# [docs:sadda.align.TimedPhone]
@dataclass(frozen=True)
class TimedPhone:
    """One aligned phone with its time span and (optional) confidence.

    ``score`` is the aligner's per-phone confidence where it provides one (the
    neural DP does); it is ``None`` for backends that don't emit one (e.g. MFA's
    TextGrid output).
    """

    label: str
    start_seconds: float
    end_seconds: float
    score: Optional[float] = None


# [docs:sadda.align.TimedWord]
@dataclass(frozen=True)
class TimedWord:
    """One aligned word: its span and the phones inside it."""

    text: str
    start_seconds: float
    end_seconds: float
    phones: tuple[TimedPhone, ...]


# [docs:sadda.align.TimedSyllable]
@dataclass(frozen=True)
class TimedSyllable:
    """One syllable: its span and the phones inside it.

    ``label`` is the syllable's phone labels concatenated (e.g. ``"loʊ"``).
    Derived from the Phone tier by rule — see :func:`syllabify`.
    """

    label: str
    start_seconds: float
    end_seconds: float
    phones: tuple[TimedPhone, ...]


# [docs:sadda.align.Alignment]
@dataclass(frozen=True)
class Alignment:
    """A forced alignment: aligned words and the flat phone sequence."""

    words: tuple[TimedWord, ...]
    phones: tuple[TimedPhone, ...]


# [docs:sadda.align.tokenize]
def tokenize(ipa: str, vocab: Mapping[str, int]) -> list[int]:
    """Greedy longest-match an IPA string into ``vocab`` class ids.

    At each position, the longest vocab key that matches wins (so ``dʒ`` beats
    ``d`` when both are in the vocab). Whitespace is skipped. Raises
    :class:`ValueError` naming the offending substring if no key matches — which
    is how a phone the acoustic model doesn't cover surfaces.
    """
    keys = sorted((k for k in vocab if k), key=len, reverse=True)
    out: list[int] = []
    i, n = 0, len(ipa)
    while i < n:
        if ipa[i].isspace():
            i += 1
            continue
        for k in keys:
            if ipa.startswith(k, i):
                out.append(vocab[k])
                i += len(k)
                break
        else:
            raise ValueError(f"phone not in acoustic-model vocab at {ipa[i:]!r}")
    return out


def _vad_silence_mask(
    audio: np.ndarray, sample_rate: int, n_frames: int, frame_rate: float
) -> list[bool]:
    """Per-emission-frame silence mask from Silero VAD (`sadda.ml`).

    Runs VAD, then marks every frame whose centre time falls outside a detected
    speech region as silence.
    """
    import sadda

    audio_obj = sadda.Audio.from_samples(audio, sample_rate, channels=1)
    segments = sadda.ml.speech_segments(audio_obj)
    mask = [True] * n_frames
    for start_s, end_s in segments:
        lo = max(0, int(start_s * frame_rate))
        hi = min(n_frames, int(round(end_s * frame_rate)))
        for f in range(lo, hi):
            mask[f] = False
    return mask


def _silence_params(
    detector: Optional[str],
    min_silence_seconds: float,
    frame_rate: float,
    audio: np.ndarray,
    sample_rate: int,
    n_frames: int,
) -> tuple[int, Optional[list[bool]]]:
    if detector is None:
        return 0, None
    if detector == "blank":
        frames = round(min_silence_seconds * frame_rate) if min_silence_seconds > 0 else 0
        return max(0, frames), None
    if detector == "vad":
        return 0, _vad_silence_mask(audio, sample_rate, n_frames, frame_rate)
    raise ValueError(f"unknown detector {detector!r}; use 'blank', 'vad', or None")


# [docs:sadda.align.align]
@provisional
def align(
    audio: np.ndarray,
    sample_rate: int,
    transcript: str,
    *,
    model: AcousticModel,
    voice: str = "en-us",
    detector: Optional[str] = "blank",
    min_silence_seconds: float = 0.20,
) -> Alignment:
    """Force-align ``transcript`` to ``audio`` with ``model``.

    Phonemizes the transcript (espeak-ng, ``voice``), gets per-frame posteriors
    from ``model``, tokenizes each word against the model vocab, runs the CTC
    forced-align DP, and returns time-aligned words and phones.

    Silence (``detector``): ``"blank"`` (default) marks CTC-blank runs at least
    ``min_silence_seconds`` long as silence; ``"vad"`` uses Silero VAD
    (:mod:`sadda.ml`); ``None`` disables it. Detected silence becomes
    **empty-labeled intervals** — the Word and Phone results stay contiguous (a
    full partition of the recording), with pauses and edge silence left empty
    rather than absorbed into neighbouring words.

    The ``min_silence_seconds`` default (0.20 s) is grounded in the pause
    literature: above typical stop-closure durations, between Praat's 0.1 s
    silence-detector default (`To TextGrid (silences)`,
    https://www.fon.hum.uva.nl/praat/manual/Sound__To_TextGrid__silences____.html)
    and Goldman-Eisler's (1968) 0.25 s articulatory-vs-hesitation-pause boundary
    (*Psycholinguistics: Experiments in Spontaneous Speech*, Academic Press — no
    stable weblink available).
    """
    audio = np.asarray(audio, dtype=np.float32)
    utt = phonemize(transcript, voice=voice)
    em = model.emissions(audio, sample_rate)
    id_to_phone = {i: p for p, i in em.vocab.items()}

    target: list[int] = []
    word_of_token: list[int] = []
    word_texts: list[str] = []
    for wi, w in enumerate(utt.words):
        word_texts.append(w.text)
        toks = tokenize(w.ipa, em.vocab)
        target.extend(toks)
        word_of_token.extend([wi] * len(toks))
    if not target:
        raise ValueError("transcript produced no phones to align")

    log_probs = np.asarray(em.log_probs, dtype=np.float32)
    fr = float(em.frame_rate)
    min_silence_frames, silence_mask = _silence_params(
        detector, min_silence_seconds, fr, audio, sample_rate, log_probs.shape[0]
    )
    spans = _native.forced_align(
        log_probs,
        target,
        blank=em.blank_id,
        min_silence_frames=min_silence_frames,
        silence_mask=silence_mask,
    )

    # Phone tier: contiguous; silence spans carry an empty label.
    phones = tuple(
        TimedPhone(
            label="" if is_sil else id_to_phone.get(label, str(label)),
            start_seconds=sf / fr,
            end_seconds=ef / fr,
            score=score,
        )
        for (_tok, label, sf, ef, score, is_sil) in spans
    )

    # Word tier: a contiguous partition. Each word spans its phones; pauses
    # between/around words are empty-labeled word intervals.
    bounds: dict[int, list[float]] = {}
    per_word_phones: dict[int, list[TimedPhone]] = {}
    for (tok, _label, sf, ef, _score, is_sil), ph in zip(spans, phones):
        if is_sil or tok >= len(word_of_token):
            continue
        wi = word_of_token[tok]
        b = bounds.setdefault(wi, [sf / fr, ef / fr])
        b[0], b[1] = min(b[0], sf / fr), max(b[1], ef / fr)
        per_word_phones.setdefault(wi, []).append(ph)

    duration = phones[-1].end_seconds if phones else 0.0
    words: list[TimedWord] = []
    prev_end = 0.0
    for wi, text in enumerate(word_texts):
        if wi not in bounds:
            continue
        w_start, w_end = bounds[wi]
        if w_start > prev_end + 1e-9:
            words.append(TimedWord(text="", start_seconds=prev_end, end_seconds=w_start, phones=()))
        words.append(
            TimedWord(text=text, start_seconds=w_start, end_seconds=w_end, phones=tuple(per_word_phones[wi]))
        )
        prev_end = w_end
    if duration > prev_end + 1e-9:
        words.append(TimedWord(text="", start_seconds=prev_end, end_seconds=duration, phones=()))

    return Alignment(words=tuple(words), phones=phones)


# [docs:sadda.align.import_alignment]
@provisional
def import_alignment(
    project,
    bundle_id: int,
    alignment: Alignment,
    *,
    word_tier: str = "words",
    phone_tier: str = "phones",
    hierarchical: bool = True,
) -> tuple[int, int]:
    """Write an :class:`Alignment` into a bundle as Word + Phone interval tiers.

    Backend-agnostic — takes the output of either the neural :func:`align` or the
    MFA :func:`sadda.align.mfa.mfa_align`, so "align, then annotate/edit" is one
    step. Creates two interval tiers on ``bundle_id``; when ``hierarchical`` the
    Phone tier is a child of the Word tier and each phone links to the word
    interval that contains it (words partition the recording contiguously, so
    every phone maps to exactly one).

    Empty labels — the neural aligner's *imputed* silence — are written as
    unlabeled intervals; *modeled* silence keeps its label (e.g. MFA
    ``sil``/``sp``), so the two silence kinds stay distinguishable on the tier.
    Returns ``(word_tier_id, phone_tier_id)``.
    """
    wt = project.add_tier(bundle_id, word_tier, "interval")
    pt = project.add_tier(
        bundle_id, phone_tier, "interval", parent_id=wt if hierarchical else None
    )
    # Word intervals; remember each span + id so phones can parent to their word.
    word_spans: list[tuple[float, float, int]] = []
    for w in alignment.words:
        wid = project.add_interval(
            wt, w.start_seconds, w.end_seconds, label=w.text or None
        )
        word_spans.append((w.start_seconds, w.end_seconds, wid))

    def _containing_word(mid: float) -> Optional[int]:
        for start, end, wid in word_spans:
            if start <= mid < end:
                return wid
        return word_spans[-1][2] if word_spans else None

    for p in alignment.phones:
        parent = (
            _containing_word((p.start_seconds + p.end_seconds) / 2.0)
            if hierarchical
            else None
        )
        project.add_interval(
            pt,
            p.start_seconds,
            p.end_seconds,
            label=p.label or None,
            parent_annotation_id=parent,
        )
    return wt, pt


# [docs:sadda.align.align_auto]
@provisional
def align_auto(
    audio: np.ndarray,
    sample_rate: int,
    *,
    model: AcousticModel,
    asr_backend: Optional[object] = None,
    language: Optional[str] = None,
    voice: str = "en-us",
    detector: Optional[str] = "blank",
    min_silence_seconds: float = 0.20,
) -> Alignment:
    """Recognize a transcript with ASR, then force-align it — the no-transcript path.

    For speech with no transcript to start from (unprompted conversational /
    naturalistic recordings): runs :func:`sadda.asr.transcribe` (default
    faster-whisper, needs ``pip install "sadda[asr]"``) to get the words, then
    :func:`align`. ``asr_backend`` overrides the recognizer; ``language`` fixes
    the spoken language (else it's detected).

    ``voice`` is the espeak-ng G2P voice used for alignment — set it to match the
    spoken language (auto-deriving it from the recognized language is a planned
    refinement). For a review-before-align workflow, call
    :func:`sadda.asr.transcribe` and :func:`align` separately instead.
    """
    from sadda import asr  # lazy: keeps sadda.align importable without the [asr] extra

    transcript = asr.transcribe(
        audio, sample_rate, backend=asr_backend, language=language
    ).text
    if not transcript:
        raise ValueError("ASR produced an empty transcript; nothing to align")
    return align(
        audio,
        sample_rate,
        transcript,
        model=model,
        voice=voice,
        detector=detector,
        min_silence_seconds=min_silence_seconds,
    )


# [docs:sadda.align.syllabify]
@provisional
def syllabify(alignment: Alignment) -> tuple[TimedSyllable, ...]:
    """Derive a Syllable tier from an :class:`Alignment` (SSP + Maximal Onset).

    Syllabifies each **word's** phones — syllabification is word-internal, so
    pauses and empty/silence intervals between words are not part of any
    syllable (empty-labelled words are skipped). Each syllable spans its phones
    and is labelled with their concatenation. Works on any `Alignment` (neural
    or MFA). See :mod:`sadda_engine::syllable` for the rule and its limitations
    (universal sonority scale, no per-language onset-legality table yet).
    """
    out: list[TimedSyllable] = []
    for w in alignment.words:
        if not w.text:  # empty-labelled pause/silence word — nothing to syllabify
            continue
        real = [p for p in w.phones if p.label]
        if not real:
            continue
        for start, end in _native.syllabify([p.label for p in real]):
            group = real[start:end]
            out.append(
                TimedSyllable(
                    label="".join(p.label for p in group),
                    start_seconds=group[0].start_seconds,
                    end_seconds=group[-1].end_seconds,
                    phones=tuple(group),
                )
            )
    return tuple(out)
