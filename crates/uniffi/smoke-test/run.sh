#!/usr/bin/env bash
# Phase 0 UniFFI smoke test runner.
#
# Builds the Rust cdylib + bindgen tool, regenerates the Swift bindings against
# the freshly-built library, compiles the Swift smoke-test program with those
# bindings, and runs it. The successful run prints the engine version through
# the Rust ↔ Swift bridge.
#
# Requires: a `swift` binary on PATH (e.g. via swiftly). The Rust side has no
# system dependencies beyond what the workspace already needs.
#
# Run from the repo root:
#   crates/uniffi/smoke-test/run.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

CRATE=sadda-uniffi
LIB=target/debug/libsadda_uniffi.so
GEN=crates/uniffi/smoke-test/generated
SWIFT_SRC=crates/uniffi/smoke-test/SmokeTest.swift
OUT=crates/uniffi/smoke-test/smoke

echo "==> cargo build $CRATE"
cargo build -p "$CRATE"

echo "==> regenerate Swift bindings from $LIB"
rm -rf "$GEN"
mkdir -p "$GEN"
./target/debug/uniffi-bindgen generate \
    --library "$LIB" \
    --language swift \
    --out-dir "$GEN" \
    --no-format

echo "==> swiftc compile smoke-test"
swiftc \
    -import-objc-header "$GEN/sadda_uniffiFFI.h" \
    -L target/debug \
    -lsadda_uniffi \
    "$GEN/sadda_uniffi.swift" \
    "$SWIFT_SRC" \
    -o "$OUT"

echo "==> running smoke test"
LD_LIBRARY_PATH="$ROOT/target/debug:${LD_LIBRARY_PATH:-}" "$OUT"
