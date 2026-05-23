# sadda.live

Live audio recording with streaming pitch / formants / intensity /
meter subscribers. PROVISIONAL tier — may change in minor versions
after a deprecation cycle.

See the [quickstart](../quickstart.md#live-recording) for a worked
example. Design rationale lives in the
[2026-05-22 DEVLOG entry "Live recording (E1)"](https://github.com/sadda-speech/sadda/blob/main/DEVLOG.md).

::: sadda.live.start_session

::: sadda.live.list_input_devices

::: sadda.live.default_input_device

::: sadda.live.LiveSession
    options:
      members:
        - start
        - stop
        - commit
        - discard
        - on_meter
        - on_pitch
        - on_intensity
        - on_formants
        - frames_written
        - dropped_samples
        - duration_seconds
        - in_progress_dir
