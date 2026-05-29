# Validation & golden references

sadda's DSP and clinical measures are validated against **authoritative
external references** wherever one exists — the reference implementation
or tool that defines the method — not just against our own expectations.
This is a deliberate part of how we earn trust in the numbers.

**You do not need any of these tools to run the test suite.** Every golden
is a small committed file (a `.tsv`/`.wav` of reference values), so
`cargo test` and CI run fully offline against them. The external tools are
only needed to **regenerate** a golden — e.g. when adding a method or
deliberately revising a fixture. Each golden lives next to a script that
documents exactly how it was produced, so the provenance is auditable and
a contributor can reproduce it.

| Area | Reference (the authority) | Tool to regenerate | Location |
|------|---------------------------|--------------------|----------|
| Jitter / shimmer / HNR / CPPS / LTAS | **Praat** 6.x | `praat` | `tests/clinical/praat/` |
| Boersma pitch tracker | **Praat** `To Pitch (ac)` | `praat` | `tests/dsp/praat/` |
| YIN / pYIN pitch | **librosa** | `librosa` (Python) | `tests/dsp/librosa/` |
| SWIPE' pitch | **Camacho's own dissertation MATLAB**, run under **Octave** | `octave` (base, no signal pkg) | `tests/dsp/swipe/` |
| Whisper log-mel front end | **OpenAI Whisper** (`audio.py` + its filterbank) | `torch` + `numpy` (Python) | `tests/dsp/whisper/` |

Notes for contributors:

- **Prefer the author's own code as the reference.** For SWIPE' we run
  Camacho's verbatim `swipep` under Octave rather than trusting a
  re-implementation — and that cross-check earned its keep: it caught a
  1-based-vs-0-based window-index bug in our first port. When a method has
  a canonical reference implementation, use it.
- **Cross-validate across families too.** Independent algorithms that
  should agree on clean input (e.g. SWIPE' vs YIN vs Boersma on a harmonic
  tone) are checked against each other in the in-module unit tests — a
  cheap, tool-free guard that complements the goldens.
- **Document deviations.** Where we knowingly differ from a reference (a
  window definition, an interpolation method), the doc comment says so and
  the golden tolerance reflects it. We aim for "faithful, or honestly
  named."
- Regeneration commands are in the header of each `make_*`/`run_*` script.
