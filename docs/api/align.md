# sadda.align

Phone-level forced alignment: time-align a transcript to audio, producing Word
and Phone results with IPA labels. PROVISIONAL tier.

For a task-oriented walk-through, see the [Forced alignment](../align.md) guide.

Requires the `espeak-ng` system binary (G2P) and `pip install "sadda[align]"`
(ONNX Runtime + `huggingface_hub`); the acoustic model is fetched from the Hub on
first use.

## Alignment

::: sadda.align.align

::: sadda.align.Alignment

::: sadda.align.TimedWord

::: sadda.align.TimedPhone

::: sadda.align.tokenize

## Grapheme-to-phoneme (espeak-ng)

::: sadda.align.phonemize

::: sadda.align.Utterance
    options:
      members:
        - phones

::: sadda.align.Word

::: sadda.align.split_phones

::: sadda.align.strip_stress

## Acoustic model

### The `AcousticModel` protocol

Any object with an `emissions(audio, sample_rate)` method returning
[`Emissions`](#sadda.align.Emissions) is an acoustic model — a structural,
runtime-checkable protocol, so no subclassing is required (a custom model or a
test mock both qualify).

```python
class AcousticModel(Protocol):
    def emissions(self, audio: np.ndarray, sample_rate: int) -> Emissions:
        """Per-frame CTC log-probabilities over a phone vocabulary."""
```

::: sadda.align.Emissions

::: sadda.align.Wav2Vec2EspeakModel
    options:
      members:
        - from_pretrained
        - emissions
