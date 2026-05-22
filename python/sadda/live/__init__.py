"""sadda.live — live audio recording with streaming DSP subscribers.

Provisional surface. See the 2026-05-22 E1 DEVLOG entry for the design,
the lossiness notes, and the deferred items (JACK, multi-channel DSP,
pause/resume, live waveform/spectrogram subscribers, reusable
sessions).

Typical usage::

    session = sadda.live.start_session(project, name="practice")

    @session.on_pitch
    def cb(f0_hz, voiced, t):
        if voiced:
            print(f"{t:.3f}s  f0={f0_hz:.1f}")

    session.start()
    time.sleep(5.0)
    session.stop()
    bundle_id = session.commit(project)

`start()` builds a cpal input stream against the requested device.
Headless test environments use ``session.push_samples_for_tests(...)``
instead — it pushes raw f32 samples directly into the engine's
ringbuffer, bypassing cpal entirely.
"""

from __future__ import annotations

from sadda import _native
from sadda._stability import provisional

__all__ = [
    "LiveSession",
    "default_input_device",
    "list_input_devices",
    "start_session",
]

start_session = provisional(_native.live.start_session)
list_input_devices = provisional(_native.live.list_input_devices)
default_input_device = provisional(_native.live.default_input_device)
LiveSession = _native.live.LiveSession
