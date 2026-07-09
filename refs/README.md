# Reference assets (local-only)

Local, **untracked** copies of papers, specs, and manuals that sadda cites —
kept here for convenient reference while working, never committed.

## Why untracked

Most of these are copyright and **cannot be redistributed**, so they must not
enter version control. This directory is git-ignored (`.gitignore`: `/refs/*`
with this README as the one tracked exception).

The **authoritative** reference is always the DOI or stable weblink recorded in
the code and citations (the `citation_for` registry, module `## References`
blocks, docstrings, and memory) — never a file in here. These PDFs are a local
convenience copy of something already weblinked; nothing should *depend* on a
file being present in `refs/`.

## Convention

- Drop a paper here when you want it at hand (e.g. while porting a method).
- Prefer the source's canonical filename or a `author-year-shortitle.pdf` form.
- If you add an asset, make sure the corresponding citation (with its DOI/URL)
  exists in the code/citations — the weblink is what actually matters.

## Contents (examples — not exhaustive, and not in git)

- `looze08_speechprosody.pdf` — De Looze & Hirst (2008), "Detecting changes in
  key and range for the automatic modelling and coding of intonation," Speech
  Prosody 2008 (doi:10.21437/SpeechProsody.2008-32). Basis for the two-pass
  adaptive pitch-range method.
