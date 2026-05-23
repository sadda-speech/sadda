# TextGrid round-trip lossiness

[Praat](https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html)
TextGrid files (`.TextGrid`) are intentionally simple — `IntervalTier`
and `TextTier` with a flat list of intervals or points. They don't
model tier hierarchy, reference annotations, or rich per-annotation
metadata.

This page documents what sadda preserves on a TextGrid round-trip,
what's lost, and what's recoverable via a JSON sentinel.

## API

```python
proj.export_textgrid(bundle_id, out_path)
proj.import_textgrid(in_path, bundle_id)
```

Both methods record a `processing_run` row of kind `dsp_algorithm` with
`processor_id = sadda.io.textgrid.import` / `…export`.

## Preserved

- `IntervalTier` ↔ `interval` tier
- `TextTier` ↔ `point` tier
- Per-annotation `text` (label) and time span
- Per-annotation `extra` JSON via the `{json:...}` sentinel
- Reference-tier values via a degenerate-interval encoding

## Lossy on export

- **Dense tiers** (`continuous_numeric` / `continuous_vector` /
  `categorical_sampled`) are silently skipped — TextGrid can't
  represent dense per-frame data.
- **Tier hierarchy** (`tier.parent_id`) — TextGrid is flat.
- **Annotation-level parentage** (`parent_annotation_id`) — same.
- **Tier `schema` JSON** — no TextGrid equivalent.

## Recovered via JSON sentinel

The annotation's `extra` JSON is appended to its label as
`<label> {json:<inline-json>}`. Praat displays the literal text; sadda
strips it on re-import. Plain TextGrids round-trip cleanly because the
sentinel is only added when `extra` is set.

```text
text = "h {json:{\"v\":1}}"
```

## Reference tiers

Reference tiers (`tier.type = 'reference'`) export as `IntervalTier`s
with a degenerate `[0.0, 0.001]` time span per row, and the JSON
sentinel carrying the `(target_kind, target_id)` payload. Re-import
recovers the data as interval annotations — reconstituting them as
reference annotations is a 0.1.x enhancement.

## Errors (not silent)

- Overlapping intervals on export raise `ValueError` (`IntervalTier`
  in Praat requires contiguous intervals).
- Bundle id not found on import raises `RuntimeError`.

## See also

- The 2026-05-22 DEVLOG entry "TextGrid round-trip (D1)" for the
  design rationale (long+short text variants, the JSON sentinel
  prefix choice, the gap-padding strategy).
- [EAF lossiness](eaf.md) for the alternative format that *does*
  preserve tier hierarchy.
