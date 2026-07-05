# sadda.tts

Backend-agnostic text-to-speech: synthesize text (and whole narration
scripts) to WAV through a pluggable backend. The immediate driver is
voiceover for generated documentation; the surface is general enough for any
TTS task. PROVISIONAL tier.

For a task-oriented walk-through — narration scripts, caching, custom
backends, and localization — see the [Text-to-speech & voiceover](../tts.md)
guide.

The default backend, `espeak-ng`, requires the `espeak-ng` system binary (it
is not a pip dependency). A high-quality neural backend (Kokoro) is planned
behind a future `sadda[tts]` extra; requesting `backend="kokoro"` today raises
an actionable error.

## Synthesis

::: sadda.tts.synthesize

::: sadda.tts.synthesize_script

::: sadda.tts.ScriptResult

::: sadda.tts.SynthesisResult

## Assembly & caching

::: sadda.tts.concat_wavs

::: sadda.tts.cache_key

## Narration scripts

::: sadda.tts.Segment

::: sadda.tts.NarrationScript
    options:
      members:
        - from_texts
        - resolved_voice
        - resolved_rate

::: sadda.tts.parse_script

::: sadda.tts.load_script

## Backends

### The `TTSBackend` protocol

A backend is any object satisfying this structural (runtime-checkable)
protocol — a `name` and a `synthesize` method. No subclassing is required;
`isinstance(obj, sadda.tts.TTSBackend)` checks the shape.

```python
class TTSBackend(Protocol):
    name: str

    def synthesize(
        self,
        text: str,
        out_path: str | Path,
        *,
        voice: str | None = None,
        rate: float | None = None,
    ) -> SynthesisResult:
        """Synthesize `text` to a WAV at `out_path` and describe the result."""
```

`voice` is a backend-specific id (e.g. `"en-us"` for espeak-ng); `rate` is a
relative multiplier where `1.0` is the backend's natural rate. Both may be
`None`, meaning "use the backend default". See the
[voiceover guide](../tts.md#writing-your-own-backend) for a worked example.

::: sadda.tts.EspeakNgBackend
    options:
      members:
        - synthesize

::: sadda.tts.get_backend

::: sadda.tts.list_backends

::: sadda.tts.register_backend
