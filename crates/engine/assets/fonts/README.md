# Bundled fonts

## Doulos SIL (Regular) — `DoulosSIL-Regular.ttf`

- **Source:** SIL Global, <https://software.sil.org/doulos/> ·
  release v7.000, <https://github.com/silnrsi/font-doulos>
- **License:** SIL Open Font License 1.1 (see `DoulosSIL-OFL.txt`).
  Reserved Font Names: "Doulos" and "SIL".
- **Why bundled:** the figure exporter (`io::figure`) embeds this font in
  exported SVG/PDF figures so IPA tier labels and axis text render identically
  everywhere, without the viewer needing the font installed. Doulos SIL is the
  phonetics-standard IPA reference face (and the specTeX style baseline).

The OFL permits redistribution and embedding; it forbids selling the font on
its own and using the Reserved Font Names on a modified version. We redistribute
the font unmodified and embed it in generated documents — both permitted.
