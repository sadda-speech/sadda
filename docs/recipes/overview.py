# sadda documentation figures — Phase 1 ("anatomy of the app").
#
# Regenerate with `just docs-images`. Each `doc.shot(...)` renders one PNG
# headlessly through the *real* app (egui_kittest + wgpu), so the images can't
# drift from what users see. Paths are relative to the repo root.
#
# Source clip: a ~6s CC0 recording ("…should I make a picture of the app?…"),
# chosen for its clear question-vs-statement intonation and varied vowels.

import sadda.doc as doc

AUDIO = "docs/recipes/assets/demo.wav"
TEXTGRID = "docs/recipes/assets/demo.TextGrid"
OUT = "docs/assets/generated"

# A1 — the hero: as much of the app as one figure can show — the whole window,
# a full measure stack (waveform, spectrogram, f0, formants, intensity, MFCC),
# and an annotation tier, light theme. Shows both the *signal analysis* and the
# *corpus/annotation* sides of sadda at a glance.
HERO_LANES = ["waveform", "spectrogram", "f0", "formants", "intensity", "mfcc", "tier_strip"]
doc.shot(
    audio=AUDIO, textgrid=TEXTGRID, size=(1280, 1000), theme="light",
    show=HERO_LANES, heights=[("waveform", 100)],
    capture="whole-window", to=f"{OUT}/overview.png",
)
# A1-dark — same, dark theme (for docs that follow the reader's theme).
doc.shot(
    audio=AUDIO, textgrid=TEXTGRID, size=(1280, 1000), theme="dark",
    show=HERO_LANES, heights=[("waveform", 100)],
    capture="whole-window", to=f"{OUT}/overview-dark.png",
)

# A2 — the core signal view: waveform over spectrogram, nothing else.
doc.shot(
    audio=AUDIO, size=(1100, 620), theme="light",
    show=["waveform", "spectrogram"], heights=[("waveform", 150)],
    capture="signal-column", to=f"{OUT}/signal-view.png",
)

# A3 — a clean spectrogram on its own (reading the picture).
doc.shot(
    audio=AUDIO, size=(1100, 560), theme="light",
    show=["spectrogram"],
    capture="spectrogram", to=f"{OUT}/spectrogram.png",
)

# A4 — the f0 (pitch) track — the question/statement intonation shows here.
# The spectrogram is the flex pane, so the f0 lane gets an explicit height.
doc.shot(
    audio=AUDIO, size=(1100, 820), theme="light",
    show=["waveform", "spectrogram", "f0"],
    heights=[("waveform", 100), ("f0", 220)],
    capture="signal-column", to=f"{OUT}/pitch-contour.png",
)

# A5 — formant tracks below the spectrogram.
doc.shot(
    audio=AUDIO, size=(1100, 820), theme="light",
    show=["spectrogram", "formants"], heights=[("formants", 260)],
    capture="signal-column", to=f"{OUT}/formant-tracks.png",
)

# A6 — the intensity contour.
doc.shot(
    audio=AUDIO, size=(1100, 560), theme="light",
    show=["waveform", "intensity"], heights=[("waveform", 180), ("intensity", 200)],
    capture="signal-column", to=f"{OUT}/intensity.png",
)

# A7 — the MFCC heatmap.
doc.shot(
    audio=AUDIO, size=(1100, 760), theme="light",
    show=["spectrogram", "mfcc"], heights=[("mfcc", 240)],
    capture="signal-column", to=f"{OUT}/mfcc.png",
)

# A8 — the full measure stack (richness in one figure).
doc.shot(
    audio=AUDIO, size=(1100, 900), theme="light",
    show=["waveform", "spectrogram", "f0", "formants", "intensity", "mfcc"],
    heights=[("waveform", 110)],
    capture="signal-column", to=f"{OUT}/measure-stack.png",
)
