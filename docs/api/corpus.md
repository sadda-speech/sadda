# sadda.corpus

The corpus surface — projects, bundles, tiers, annotations. STABLE
tier: won't break across minor versions.

Project loaders live at the top of the `sadda` package:

::: sadda.new_project
    options:
      show_root_heading: true

::: sadda.open_project
    options:
      show_root_heading: true

## Project

::: sadda.Project
    options:
      show_root_heading: true
      members:
        - root
        - name
        - add_bundle
        - bundles
        - load_audio
        - add_tier
        - tiers
        - get_tier
        - add_interval
        - intervals
        - add_point
        - points
        - add_reference
        - references_for
        - tier_rows
        - query
        - write_continuous_numeric
        - write_continuous_vector
        - write_categorical_sampled
        - load_continuous_numeric
        - load_continuous_vector
        - load_categorical_sampled
        - import_textgrid
        - export_textgrid
        - import_eaf
        - export_eaf
        - commit_recording

## Data types

::: sadda.Audio

::: sadda.Bundle

::: sadda.Tier

::: sadda.Interval

::: sadda.Point

::: sadda.Reference

::: sadda.DerivedSignal

::: sadda.Speaker

::: sadda.Session
