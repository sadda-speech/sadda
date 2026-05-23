# sadda.dsp

Pure-function DSP toolkit. Every function takes NumPy `float32` audio
+ a sample rate and returns NumPy or dataclass results. No corpus
dependency. STABLE tier.

::: sadda.dsp
    options:
      show_root_heading: false
      members:
        - hann
        - hamming
        - blackman
        - gaussian
        - kaiser
        - stft
        - spectrogram
        - intensity
        - f0
        - voiced_pitch
        - formants
        - mfcc
        - FormantFrame
