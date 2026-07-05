# sadda task runner — https://github.com/casey/just
#
# Install once:  cargo install just   (or: brew install just / apt install just)
# List recipes:  just            (alias for `just --list`)
# Pre-commit:    just gate       (mirrors .github/workflows/gate.yml locally)
#
# The recipes below reproduce the CI gate step-for-step so a green `just gate`
# means a green CI. The gate is a reusable workflow (.github/workflows/gate.yml)
# that CI *and* both release workflows (release.yml, app-release.yml) call — so
# a green gate is also what hard-gates a publish. Keep this file and gate.yml in
# sync — if you add a step there, add it here too.

# crates/script-engine embeds CPython via pyo3 auto-initialize, so the Rust
# test binaries and stub_gen dlopen libpython at runtime. On CI that's the
# system python, already on the loader path. Locally it lives in conda's lib
# dir, which ~/.bashrc deliberately keeps OFF the global LD_LIBRARY_PATH (it
# shadows system OpenSSL/ncurses) and scopes per-command via `with_conda_libs`.
# We do the same here: prepend libpython's dir for the gate's processes only.
# Defaults to $CONDA_PREFIX/lib (so it's not a hardcoded path); override with
# SADDA_PYLIB=/some/dir. Empty when neither is set (CI) → leaves LD_LIBRARY_PATH
# untouched. See crates/script-engine/README.md.
pylib := env_var_or_default("SADDA_PYLIB", if env_var_or_default("CONDA_PREFIX", "") != "" { env_var("CONDA_PREFIX") / "lib" } else { "" })
export LD_LIBRARY_PATH := if pylib == "" { env_var_or_default("LD_LIBRARY_PATH", "") } else { pylib + if env_var_or_default("LD_LIBRARY_PATH", "") != "" { ":" + env_var("LD_LIBRARY_PATH") } else { "" } }

# Show the recipe list (default when you type bare `just`).
default:
    @just --list

# ── The gate ────────────────────────────────────────────────────────────────
# Order matches gate.yml: fmt → clippy → build → test → download-feature →
# stub-drift → pytest. Fails fast on the first broken step.

# Full pre-commit / pre-push check — a green run here == green CI.
gate: fmt-check clippy build build-release test test-download stubs pytest
    @echo ""
    @echo "✅ gate passed — fmt · clippy · build · test · download · stubs · pytest"

# ── Individual gate steps (run any in isolation) ────────────────────────────

# cargo fmt --check (formatting only; does not modify files).
fmt-check:
    cargo fmt --all -- --check

# This is the warning gate: clippy compiles every target, so it catches plain
# rustc warnings too. (CI also sets RUSTFLAGS=-D warnings on the bare `build`
# step; we deliberately omit that here so plain `cargo build` in your shell
# shares the same target/ cache instead of thrashing it. clippy -D warnings
# already covers the intent.)

# Clippy across the whole workspace, all targets, warnings = errors.
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Build the whole workspace, all targets.
build:
    cargo build --workspace --all-targets

# Release compile-check of the app. The gate (and CI) otherwise build only in
# debug, so `debug_assertions`-gated egui APIs (e.g. `set_debug_on_hover`) that
# vanish in release slip through and break the app-release workflow's
# `--release` build. `check` (no codegen) keeps this quick.
build-release:
    cargo check --release -p sadda-app

# Run the full Rust test suite.
test:
    cargo test --workspace

# Micro-benchmarks (divan), behind the off-by-default `spike` feature so the
# throwaway spike code never ships. Verifies compositional==production first,
# then runs the perf bench (fused vs naive-materialised vs streaming), with
# per-run allocation stats. See DEVLOG.
bench:
    cargo test -p sadda-engine --features spike compositional_variants_match_production
    cargo bench -p sadda-engine --features spike

# The `download` feature (E12, network model fetch) is enabled by no workspace
# member, so the default --workspace passes don't compile it. Network/ORT-gated
# tests skip without SADDA_NET_TESTS / ORT_DYLIB_PATH.

# Check + test the `download` feature explicitly (not covered by --workspace).
test-download:
    cargo clippy -p sadda-engine --features download --all-targets -- -D warnings
    cargo test -p sadda-engine --features download

# Matches CI's "stubs no-drift" check. Run `just stub-gen` to regenerate +
# accept the new stubs.

# Regenerate the Python type stubs and fail if they differ from what's committed.
stubs: stub-gen
    @git diff --exit-code python/sadda/_native/__init__.pyi \
      || (echo "::error:: python/sadda/_native/__init__.pyi is stale — run 'just stub-gen' and commit." && exit 1)

# Regenerate the type stubs in place (no drift check). Commit the result.
# Unlike CI we don't wrap this in `uv run`: stub_gen is a pure Rust binary that
# reads pyo3 metadata (no venv wheel needed), so building it against the same
# conda python as the other recipes keeps one ABI in target/ and avoids a
# 3.11-vs-3.12 libpython split. Stub content is python-version-independent.
stub-gen:
    cargo run --bin stub_gen

# Rebuild the sadda extension into .venv from the current Rust. CI does this
# via a fresh `uv sync`; `uv run` alone won't rebuild on Rust-source changes,
# so without this pytest would silently test a STALE extension (a Rust change
# to the Python surface could pass/fail wrongly). CONDA_PREFIX is unset for the
# build because maturin refuses when both it and VIRTUAL_ENV are set; conda's
# libpython stays available via the LD_LIBRARY_PATH exported above.
develop:
    env -u CONDA_PREFIX uv run maturin develop --quiet

# Python test suite (library + docs examples) — rebuilds the extension first.
pytest: develop
    uv run pytest python/tests/ tools/docs/

# Regenerate documentation images headlessly (S7). Drives the real app offscreen
# via egui_kittest + wgpu — the same stack users see, so images can't drift.
# Needs a software-Vulkan (lavapipe) adapter; the tests auto-detect the ICD, or
# set VK_ICD_FILENAMES + WGPU_BACKEND=vulkan yourself (CI does). Recipes live in
# `crates/app/src/doc_render.rs`; rendered images land under `target/doc-render/`.
docs-images:
    cargo test -p sadda-app --bins doc_render -- --ignored --nocapture --test-threads=1

# Refresh the committed snapshot goldens after an *intended* UI change (S8).
# Review the resulting image diff before committing — this is the deliberate
# "yes, the docs figure should change" step.
docs-images-update:
    UPDATE_SNAPSHOTS=1 cargo test -p sadda-app --bins doc_render -- --ignored --test-threads=1

# ── Convenience ─────────────────────────────────────────────────────────────

# Auto-format (the mutating counterpart to fmt-check).
fmt:
    cargo fmt --all
