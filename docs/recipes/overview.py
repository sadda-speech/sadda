# Example sadda documentation recipe.
#
# Regenerate the images with `just docs-images`. Each `doc.shot(...)` renders one
# PNG headlessly through the *real* app (egui_kittest + wgpu), so the images
# can't drift from what users actually see. Paths are relative to the repo root:
# `audio=`/`project=` are inputs, `to=` is the output (kept under the git-ignored
# `target/` for now; point it at `docs/...` once we commit demo images).
#
# `audio=` builds a throwaway one-bundle project from a WAV, so the recipe stays
# self-contained (no project database to commit). A small synthetic fixture
# stands in for a demo clip until a clean-licensed speech sample is vendored.

import sadda.doc as doc

AUDIO = "crates/engine/tests/clinical/fixtures/hnr_high_120hz.wav"

# Waveform + spectrogram, light theme, waveform pinned thin — the canonical
# "here's the signal view" figure.
doc.shot(
    audio=AUDIO,
    size=(1200, 760),
    theme="light",
    show=["waveform", "spectrogram"],
    heights=[("waveform", 130)],
    capture="signal-column",
    to="target/doc-render/example-overview.png",
)
