# EAF round-trip lossiness

ELAN's [EAF format](https://www.mpi.nl/tools/elan/EAF_Annotation_Format_3.0_and_ELAN.pdf)
is substantially richer than Praat TextGrid — tier hierarchy via
`PARENT_REF`, multiple linguistic-type stereotypes
(`Time_Subdivision`, `Symbolic_Association`, …), controlled
vocabularies, language tags, license metadata.

sadda preserves the parts that map cleanly to its corpus data model
and explicitly drops the rest.

## API

```python
proj.export_eaf(bundle_id, out_path)
proj.import_eaf(in_path, bundle_id)
```

Both methods record a `processing_run` row of kind `dsp_algorithm`
with `processor_id = sadda.io.eaf.import` / `…export`. Tier hierarchy
**is preserved** on round-trip via `PARENT_REF` ↔ `tier.parent_id` —
the headline difference from TextGrid.

## Preserved

- Tier hierarchy via `PARENT_REF` ↔ `tier.parent_id`
- Annotation values (label + `extra` via the JSON sentinel)
- Time alignment (millisecond precision)
- Reference annotations (`SYMBOLIC_ASSOCIATION` linguistic type ↔
  sadda `reference` tier)
- Point tiers via a degenerate `[t, t + 1ms]` alignable encoding
  (recovered on import via the `end - start ≤ 2ms` heuristic)

## Lossy on round-trip

ELAN files commonly carry metadata that doesn't fit sadda's model.
The following are **dropped silently**:

- `CONTROLLED_VOCABULARY` (CV_ENTRY, CV_REF)
- `LANGUAGE` / `LOCALE` elements
- `LICENSE`, `AUTHOR`, `DATE` attributes
- `EXTERNAL_REF`, `LEXICON_REF` references
- `REF_LINK_SET`
- Stereotypes beyond the three named (`Time_Subdivision`,
  `Symbolic_Subdivision`, `Symbolic_Association`) — others are
  simplified
- Original annotation IDs (fresh `a<N>` IDs are minted on export)
- Media-file path metadata in `<HEADER>` — sadda writes a placeholder
  pointing at the bundle's audio

Preserving this metadata opaquely would require a per-tier
`extra_xml` column (schema migration) plus opaque-XML retention
logic. Tracked as a future enhancement; pending real-user demand.

## Recovered via JSON sentinel

The annotation's `extra` JSON is appended as
`<label> {json:<inline-json>}` (same scheme as TextGrid). On reference
tiers, the sentinel additionally carries the
`(target_kind, target_id)` payload so reference annotations
round-trip losslessly between sadda projects.

## XML entities

The writer XML-escapes inner `"` characters in annotation values
(via `quick_xml`'s default escape), so a JSON sentinel like
`{"v":1}` is emitted as `{&quot;v&quot;:1}`. The parser stitches the
escaped entities back together — without it the quotes would silently
disappear on round-trip.

## FORMAT version

- Write: emits `FORMAT="2.8"` (widely supported by ELAN 5.0+).
- Read: permissive — accepts 2.7 / 2.8 / 3.0. EAF 3.0-only features
  we don't use (external CV references) are ignored.

## See also

- The 2026-05-22 DEVLOG entry "EAF round-trip (D2)" for the full
  design rationale, the point-tier heuristic justification, and the
  `cardinality = "none"` semantics fix that enables tier-hierarchy
  recovery without annotation-level parentage.
- [TextGrid lossiness](textgrid.md) for the simpler alternative.
