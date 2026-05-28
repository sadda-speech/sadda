# Third-Party Notices

The sadda desktop application's release bundles include third-party
software whose licenses require attribution and reproduction of the
license text. This file lists each such component, its origin, and its
license. See `LICENSE-APACHE` / `LICENSE-MIT` for sadda's own licensing.

The application's source tree also depends on a much larger set of Rust
and Python libraries via Cargo / pip — those are not redistributed in
binary form inside our release archives and are covered by their own
upstream license files in the dependency cache. The components below
are the ones whose binaries we ship.

---

## ONNX Runtime

- **Version**: 1.22.0
- **Source**: <https://github.com/microsoft/onnxruntime>
- **Release**: <https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0>
- **License**: MIT
- **Shipped as**: `onnxruntime/libonnxruntime.so.1.22.0` (Linux x64) /
  `onnxruntime/libonnxruntime.1.22.0.dylib` (macOS arm64) /
  `onnxruntime/onnxruntime.dll` (Windows x64), with the upstream
  `LICENSE` file beside it in each release archive.

The full upstream license text is reproduced verbatim from
<https://github.com/microsoft/onnxruntime/blob/v1.22.0/LICENSE>:

```
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Silero VAD

- **Version**: 6.2.1 (`silero_vad.onnx` weights)
- **Source**: <https://github.com/snakers4/silero-vad>
- **License**: MIT
- **Shipped as**: bundled inside the sadda source tree at
  `models-bundled/silero-vad/silero_vad.onnx`, with the upstream
  `LICENSE` file alongside.

The full upstream license text is reproduced verbatim from
<https://github.com/snakers4/silero-vad/blob/master/LICENSE>:

```
MIT License

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
