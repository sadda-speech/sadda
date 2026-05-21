# sadda-script-engine

Embedded CPython runtime for sadda. Lets the desktop app run Python scripts
inside the host Rust process.

## Build requirement: a shared `libpython`

This crate uses `pyo3` with the `auto-initialize` feature, which means
**libpython is linked into the host binary** (rather than the
`extension-module` setup used by `crates/python`, where Python loads the
.so).

PyO3 picks a Python interpreter at build time via the `PYO3_PYTHON`
environment variable, defaulting to `python3` on `PATH`. That Python must
have a shared `libpython` — i.e. `sysconfig.get_config_var('LDLIBRARY')`
must end in `.so` (or `.dylib` / `.dll` on macOS / Windows), not `.a`.

Common gotcha: **Anaconda / Miniconda Python is static-only by default**
(`LDLIBRARY = libpython3.X.a`). If your `python3` on `PATH` is conda, the
build will succeed but `cargo test -p sadda-script-engine` will fail at
runtime with:

```
error while loading shared libraries: libpython3.X.so.1.0:
cannot open shared object file: No such file or directory
```

The fix is to point `PYO3_PYTHON` at a Python that ships a shared
`libpython`:

```bash
# Linux (system Python 3.10+ from apt usually has libpython3.X.so)
PYO3_PYTHON=/usr/bin/python3 cargo test -p sadda-script-engine

# macOS (Homebrew python builds with libpython3.X.dylib)
PYO3_PYTHON=$(brew --prefix python@3.12)/bin/python3.12 cargo test -p sadda-script-engine
```

To make this persistent for your dev environment, add it to your shell rc
or a workspace-local `.cargo/config.toml`:

```toml
# .cargo/config.toml (gitignored; per-developer)
[env]
PYO3_PYTHON = "/usr/bin/python3"
```

CI sets `PYO3_PYTHON=/usr/bin/python3` in the workflow environment so it
doesn't depend on whichever Python happens to be first on the runner's
`PATH`.

## Module status

Phase 0 scope only: `run_script(code) -> ScriptOutput { stdout, stderr }`
with each call running in a fresh globals namespace. The lib is
intentionally minimal until the egui script panel and engine-API exposure
work pick it up in a follow-up slice.
