"""sadda.dsp — foundational DSP toolkit.

Pure-function API over NumPy float32 arrays. Window functions, STFT,
spectrogram, intensity, and the relocated `f0` from Phase 0 all live here.
Stability tier: STABLE (per the 2026-05-18 Python API surface DEVLOG entry).

The top-level `sadda.f0` stays as a Phase-0 back-compat alias for the same
function.
"""

from __future__ import annotations

from typing import Optional

from sadda import _native
from sadda._stability import provisional, stable

__all__ = [
    "FormantFrame",
    "FormantPreset",
    "FormantsParams",
    "Ltas",
    "MfccParams",
    "MfccPreset",
    "PitchParams",
    "PitchPreset",
    "blackman",
    "builtin_formant_presets",
    "builtin_mfcc_presets",
    "builtin_pitch_presets",
    "delete_formant_preset",
    "delete_mfcc_preset",
    "delete_pitch_preset",
    "f0",
    "formant_preset",
    "formant_preset_store",
    "formant_presets",
    "formant_user_presets",
    "formants",
    "gaussian",
    "hamming",
    "hann",
    "intensity",
    "kaiser",
    "log_mel_whisper",
    "ltas",
    "mfcc",
    "mfcc_preset",
    "mfcc_preset_store",
    "mfcc_presets",
    "mfcc_user_presets",
    "pitch_preset",
    "pitch_preset_store",
    "pitch_presets",
    "pitch_user_presets",
    "save_formant_preset",
    "save_mfcc_preset",
    "save_pitch_preset",
    "spectrogram",
    "stft",
    "voiced_pitch",
    "estimate_pitch_range",
    "speaker_pitch_range_pooled",
    "speaker_pitch_ranges_shrunk",
]

hann = stable(_native.hann)
hamming = stable(_native.hamming)
blackman = stable(_native.blackman)
gaussian = stable(_native.gaussian)
kaiser = stable(_native.kaiser)
stft = stable(_native.stft)
spectrogram = stable(_native.spectrogram)
intensity = stable(_native.intensity)
f0 = stable(_native.f0)

# C2 surface.


@stable
def formants(
    audio,
    *,
    params: "Optional[FormantsParams]" = None,
    frame_size_seconds: float = 0.025,
    hop_seconds: float = 0.010,
    n_formants: int = 5,
    pre_emphasis: float = 0.97,
    lpc_order: Optional[int] = None,
    method: str = "burg",
    max_bandwidth_hz: float = 1000.0,
    min_frequency_hz: float = 50.0,
):
    """Per-frame formants via LPC + root-finding; returns a list of
    ``FormantFrame``.

    Either pass ``method`` (``burg`` (default) or ``autocorrelation``) with the
    analysis keywords, or pass ``params=`` a :class:`FormantsParams` (from a
    preset, optionally edited with ``.replace(...)``). When ``params`` is given
    it fully determines the computation and the other keywords are ignored."""
    if params is not None:
        return _native.formant_preset.compute(audio, params)
    return _native.formants(
        audio,
        frame_size_seconds=frame_size_seconds,
        hop_seconds=hop_seconds,
        n_formants=n_formants,
        pre_emphasis=pre_emphasis,
        lpc_order=lpc_order,
        method=method,
        max_bandwidth_hz=max_bandwidth_hz,
        min_frequency_hz=min_frequency_hz,
    )


# Formant parameter/preset registry (roadmap item 6). PROVISIONAL.
FormantsParams = _native.formant_preset.FormantsParams
FormantPreset = _native.formant_preset.FormantPreset


@provisional
def formant_presets(*, root: Optional[str] = None) -> "list[FormantPreset]":
    """All formant presets: the built-in reference methods (praat-burg /
    autocorrelation) followed by the user's on-disk presets."""
    return _native.formant_preset.list_all(root=root)


@provisional
def formant_user_presets(*, root: Optional[str] = None) -> "list[FormantPreset]":
    """The user's on-disk formant presets only (no built-ins)."""
    return _native.formant_preset.list_user(root=root)


@provisional
def builtin_formant_presets() -> "list[FormantPreset]":
    """The built-in authoritative formant presets (praat-burg / autocorrelation)."""
    return _native.formant_preset.builtin()


@provisional
def formant_preset(id: str, *, root: Optional[str] = None) -> "Optional[FormantPreset]":  # noqa: A002
    """The formant preset with this ``id`` (built-in or on-disk), or ``None``."""
    return _native.formant_preset.get(id, root=root)


@provisional
def save_formant_preset(preset: "FormantPreset", *, root: Optional[str] = None) -> str:
    """Save a user formant preset to the store; returns its file path."""
    return _native.formant_preset.save(preset, root=root)


@provisional
def delete_formant_preset(id: str, *, root: Optional[str] = None) -> bool:  # noqa: A002
    """Delete the user formant preset with this ``id``. Returns ``True`` if a
    file was removed."""
    return _native.formant_preset.delete(id, root=root)


@provisional
def formant_preset_store(*, root: Optional[str] = None) -> str:
    """Filesystem path of the active formant preset store (default:
    ``~/.local/share/sadda/presets/formant/`` or the platform equivalent)."""
    return _native.formant_preset.store_root(root=root)


@stable
def voiced_pitch(
    audio,
    *,
    params: "Optional[PitchParams]" = None,
    frame_size_seconds: float = 0.030,
    hop_size_seconds: float = 0.010,
    min_freq_hz: float = 75.0,
    max_freq_hz: float = 500.0,
    method: str = "boersma",
    voicing_threshold: float = 0.45,
    range_mode: str = "manual",
):
    """Estimate f0 with a voicing decision; returns ``(times, frequencies,
    voicing)`` as three NumPy arrays.

    Either pass ``method`` (one of ``autocorrelation`` |
    ``windowed_autocorrelation`` | ``boersma`` (default) | ``yin`` | ``pyin`` |
    ``swipe``) with the common analysis keywords, or pass ``params=`` a
    :class:`PitchParams` (from a preset, optionally edited with
    ``.replace(...)``). When ``params`` is given it fully determines the
    computation and the other keywords are ignored.

    ``range_mode`` chooses how the analysis floor/ceiling are set:
    ``"manual"`` (default) uses ``min_freq_hz`` / ``max_freq_hz`` as given;
    ``"two_pass"`` adapts them to the recording (De Looze & Hirst 2008 —
    ``floor = 0.75·q25``, ``ceiling = 1.5·q75`` from a wide first pass, then
    re-track), ignoring ``min_freq_hz`` / ``max_freq_hz``. See
    :func:`estimate_pitch_range`."""
    if params is not None:
        return _native.pitch_preset.compute(audio, params)
    return _native.voiced_pitch(
        audio,
        frame_size_seconds=frame_size_seconds,
        hop_size_seconds=hop_size_seconds,
        min_freq_hz=min_freq_hz,
        max_freq_hz=max_freq_hz,
        method=method,
        voicing_threshold=voicing_threshold,
        range_mode=range_mode,
    )


@stable
def estimate_pitch_range(
    audio,
    *,
    method: str = "boersma",
    voicing_threshold: float = 0.45,
):
    """Estimate a speaker-appropriate ``(floor_hz, ceiling_hz)`` from one
    recording via the De Looze & Hirst (2008) two-pass rule, or ``None`` if the
    recording has too few voiced frames.

    Analyses f0 over a wide range, then returns ``floor = 0.75·q25`` and
    ``ceiling = 1.5·q75`` from the first/third quartiles of the voiced f0 — the
    same range :func:`voiced_pitch` derives under ``range_mode="two_pass"``.

    Reference: De Looze & Hirst (2008),
    https://doi.org/10.21437/SpeechProsody.2008-32."""
    return _native.estimate_pitch_range(
        audio, method=method, voicing_threshold=voicing_threshold
    )


@stable
def speaker_pitch_range_pooled(
    audios,
    *,
    method: str = "boersma",
    voicing_threshold: float = 0.45,
):
    """Complete-pooling speaker pitch range: pool the voiced f0 of *all* a
    speaker's recordings (``audios``, an iterable of :class:`Audio`) into one
    distribution, then return a single ``(floor_hz, ceiling_hz)`` via the De
    Looze & Hirst (2008) rule (``floor = 0.75·q25``, ``ceiling = 1.5·q75``), or
    ``None`` if the pooled f0 has too few voiced frames.

    Each recording is analysed over a wide range first so the pooled quartiles
    aren't clipped. The partial-pooling companion is
    :func:`speaker_pitch_ranges_shrunk` (empirical Bayes)."""
    return _native.speaker_pitch_range_pooled(
        list(audios), method=method, voicing_threshold=voicing_threshold
    )


@stable
def speaker_pitch_ranges_shrunk(
    audios,
    *,
    method: str = "boersma",
    voicing_threshold: float = 0.45,
):
    """Empirical-Bayes speaker pitch ranges: *partially* pool a speaker's
    recordings (``audios``, an iterable of :class:`Audio`). Each recording's
    quartiles are shrunk toward the speaker-pooled quartiles by an amount that
    grows as its voiced-frame count falls, then the De Looze & Hirst (2008) rule
    is applied per recording.

    Returns a list aligned with ``audios``: each entry is that recording's
    shrunken ``(floor_hz, ceiling_hz)``, or ``None`` if it had too few voiced
    frames. The partial-pooling companion to :func:`speaker_pitch_range_pooled`
    (sadda's own method, after Efron & Morris empirical Bayes)."""
    return _native.speaker_pitch_ranges_shrunk(
        list(audios), method=method, voicing_threshold=voicing_threshold
    )


# Pitch parameter/preset registry (roadmap item 6). PROVISIONAL.
PitchParams = _native.pitch_preset.PitchParams
PitchPreset = _native.pitch_preset.PitchPreset


@provisional
def pitch_presets(*, root: Optional[str] = None) -> "list[PitchPreset]":
    """All pitch presets: the built-in reference trackers (praat-ac / yin /
    pyin / swipe) followed by the user's on-disk presets."""
    return _native.pitch_preset.list_all(root=root)


@provisional
def pitch_user_presets(*, root: Optional[str] = None) -> "list[PitchPreset]":
    """The user's on-disk pitch presets only (no built-ins)."""
    return _native.pitch_preset.list_user(root=root)


@provisional
def builtin_pitch_presets() -> "list[PitchPreset]":
    """The built-in authoritative pitch presets (praat-ac / yin / pyin / swipe)."""
    return _native.pitch_preset.builtin()


@provisional
def pitch_preset(id: str, *, root: Optional[str] = None) -> "Optional[PitchPreset]":  # noqa: A002
    """The pitch preset with this ``id`` (built-in or on-disk), or ``None``."""
    return _native.pitch_preset.get(id, root=root)


@provisional
def save_pitch_preset(preset: "PitchPreset", *, root: Optional[str] = None) -> str:
    """Save a user pitch preset to the store; returns its file path."""
    return _native.pitch_preset.save(preset, root=root)


@provisional
def delete_pitch_preset(id: str, *, root: Optional[str] = None) -> bool:  # noqa: A002
    """Delete the user pitch preset with this ``id``. Returns ``True`` if a file
    was removed."""
    return _native.pitch_preset.delete(id, root=root)


@provisional
def pitch_preset_store(*, root: Optional[str] = None) -> str:
    """Filesystem path of the active pitch preset store (default:
    ``~/.local/share/sadda/presets/pitch/`` or the platform equivalent)."""
    return _native.pitch_preset.store_root(root=root)


@stable
def mfcc(
    audio,
    *,
    params: "Optional[MfccParams]" = None,
    frame_size_seconds: float = 0.025,
    hop_seconds: float = 0.010,
    n_mels: int = 40,
    n_mfcc: int = 13,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    method: str = "librosa",
):
    """Mel-frequency cepstral coefficients, shape ``(n_frames, n_mfcc)``.

    Two ways to specify the computation:

    - **By named method** (default): ``method`` is one of ``"librosa"``
      (default), ``"kaldi"``, or ``"praat"`` — each a faithful reproduction of
      that reference (see ``MfccMethod``). The other keyword args set the
      common analysis parameters.
    - **By full parameter set**: pass ``params=`` an :class:`MfccParams` (from a
      preset, optionally edited with ``.replace(...)``). When ``params`` is
      given it fully determines the computation and the ``method`` / ``n_mels``
      / ``frame_size_seconds`` / … keywords are ignored.

    ``f_max`` defaults to the Nyquist frequency (``sample_rate / 2``)."""
    if params is not None:
        return _native.mfcc_preset.compute(audio, params)
    return _native.mfcc(
        audio,
        frame_size_seconds=frame_size_seconds,
        hop_seconds=hop_seconds,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        f_min=f_min,
        f_max=f_max,
        method=method,
    )


# MFCC parameter/preset registry (roadmap item 3/4). The params + preset types
# and the on-disk store. PROVISIONAL — the preset schema may still change.
# Classes are re-exported raw (not `@provisional`): that decorator wraps
# `__init__`, which breaks PyO3's `__new__`-based construction — so the
# provisional tier is carried by the store functions below, as in sadda.refdist.
MfccParams = _native.mfcc_preset.MfccParams
MfccPreset = _native.mfcc_preset.MfccPreset


@provisional
def mfcc_presets(*, root: Optional[str] = None) -> "list[MfccPreset]":
    """All MFCC presets: the built-in authoritative set (librosa / kaldi /
    praat) followed by the user's on-disk presets."""
    return _native.mfcc_preset.list_all(root=root)


@provisional
def mfcc_user_presets(*, root: Optional[str] = None) -> "list[MfccPreset]":
    """The user's on-disk MFCC presets only (no built-ins)."""
    return _native.mfcc_preset.list_user(root=root)


@provisional
def builtin_mfcc_presets() -> "list[MfccPreset]":
    """The built-in authoritative MFCC presets (librosa / kaldi / praat)."""
    return _native.mfcc_preset.builtin()


@provisional
def mfcc_preset(id: str, *, root: Optional[str] = None) -> "Optional[MfccPreset]":  # noqa: A002
    """The MFCC preset with this ``id`` (built-in or on-disk), or ``None``."""
    return _native.mfcc_preset.get(id, root=root)


@provisional
def save_mfcc_preset(preset: "MfccPreset", *, root: Optional[str] = None) -> str:
    """Save a user MFCC preset to the store; returns its file path. Errors if
    the id is invalid or collides with a built-in (which are immutable)."""
    return _native.mfcc_preset.save(preset, root=root)


@provisional
def delete_mfcc_preset(id: str, *, root: Optional[str] = None) -> bool:  # noqa: A002
    """Delete the user MFCC preset with this ``id``. Returns ``True`` if a file
    was removed."""
    return _native.mfcc_preset.delete(id, root=root)


@provisional
def mfcc_preset_store(*, root: Optional[str] = None) -> str:
    """Filesystem path of the active MFCC preset store (default:
    ``~/.local/share/sadda/presets/mfcc/`` or the platform equivalent)."""
    return _native.mfcc_preset.store_root(root=root)
# Whisper-exact log-mel front end (the byte-faithful Whisper-encoder input).
log_mel_whisper = stable(_native.log_mel_whisper)
FormantFrame = stable(_native.FormantFrame)

# LTAS surface — long-term average spectrum + slope/tilt/alpha ratio.
ltas = stable(_native.ltas)
Ltas = stable(_native.Ltas)
