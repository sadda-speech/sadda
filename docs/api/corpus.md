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
        - add_bundle_split
        - bundles
        - rename_bundle
        - delete_bundle
        - load_audio
        - add_speaker
        - speakers
        - get_speaker
        - add_session
        - sessions
        - get_session
        - add_instrument
        - instruments
        - get_instrument
        - bundle_calibration
        - add_tier
        - tiers
        - get_tier
        - rename_tier
        - delete_tier
        - add_interval
        - intervals
        - add_point
        - points
        - add_reference
        - references_for
        - query
        - write_continuous_numeric
        - write_continuous_vector
        - write_categorical_sampled
        - read_continuous_numeric
        - read_continuous_vector
        - read_categorical_sampled
        - dense_path
        - derived_signal
        - import_textgrid
        - export_textgrid
        - import_eaf
        - export_eaf
        - record_processing_run
        - processing_runs
        - citations
        - extract_embeddings
        - pin_refdist
        - refdist_pins
        - remove_refdist_pin
        - audit_user
        - set_audit_user

## Data types

::: sadda.Audio

::: sadda.AudioProbe

::: sadda.Bundle

::: sadda.Tier

::: sadda.Interval

::: sadda.Point

::: sadda.Reference

::: sadda.DerivedSignal

::: sadda.Speaker

::: sadda.Session

::: sadda.Instrument

::: sadda.Calibration

::: sadda.ProcessingRun

::: sadda.Citation
