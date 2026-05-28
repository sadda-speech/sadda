# pitch_boersma.praat — generate Praat golden values for sadda's
# faithful-Boersma pitch tracker validation.
#
# Reads every *.wav under the fixtures directory and writes one TSV row
# per file with Praat's `Sound: To Pitch (ac)…` median voiced f0 (the
# faithful Boersma 1993 tracker) plus the voiced-frame count, using
# Praat's documented default parameters: time_step = 0 (auto),
# pitch_floor = 75, max_candidates = 15, very_accurate = no,
# silence_threshold = 0.03, voicing_threshold = 0.45, octave_cost = 0.01,
# octave_jump_cost = 0.35, voiced_unvoiced_cost = 0.14, pitch_ceiling
# = 600. These are also `crate::pitch::PitchConfig`'s defaults for
# `PitchMethod::Boersma`, so a same-defaults comparison is meaningful.
#
# This is the record of how the target (golden) data was produced.
# Per the 2026-05-25 clinical-validation-references entry, Praat is the
# primary reference; the resulting TSV is committed so CI never needs
# Praat. Regenerate by re-running this on the fixture WAVs.
#
# Praat 6.2.09 (February 15, 2022).
# Run from the repo root:
#   praat --run crates/engine/tests/dsp/praat/pitch_boersma.praat \
#         /home/db/Projects/sadda/crates/engine/tests/clinical/fixtures

form Boersma pitch tracking
    sentence fixtures_dir crates/engine/tests/clinical/fixtures
endform

out$ = fixtures_dir$ + "/pitch_boersma_golden.tsv"
writeFileLine: out$, "signal", tab$, "n_voiced_frames", tab$, "median_f0_hz"

# Praat `Sound: To Pitch (ac)…` defaults.
time_step = 0.0
pitch_floor = 75
max_candidates = 15
very_accurate$ = "no"
silence_threshold = 0.03
voicing_threshold = 0.45
octave_cost = 0.01
octave_jump_cost = 0.35
voiced_unvoiced_cost = 0.14
pitch_ceiling = 600

strings = Create Strings as file list: "list", fixtures_dir$ + "/*.wav"
n = Get number of strings
for i to n
    selectObject: strings
    name$ = Get string: i
    sound = Read from file: fixtures_dir$ + "/" + name$
    stem$ = name$ - ".wav"

    selectObject: sound
    pitch = To Pitch (ac): time_step, pitch_floor, max_candidates,
        ... very_accurate$, silence_threshold, voicing_threshold,
        ... octave_cost, octave_jump_cost, voiced_unvoiced_cost,
        ... pitch_ceiling
    n_voiced = Count voiced frames
    median_f0 = Get quantile: 0, 0, 0.5, "Hertz"

    appendFileLine: out$, stem$, tab$, n_voiced, tab$, median_f0

    removeObject: sound, pitch
endfor
removeObject: strings
writeInfoLine: "Wrote ", n, " rows to ", out$
