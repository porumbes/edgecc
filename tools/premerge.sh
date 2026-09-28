#!/usr/bin/env bash
# Pre-merge gate for edgecc: run before merging any feature branch.
#
#   1. normal build (build/) + full ctest
#   2. UBSan build (build-ubsan/) + full ctest
#   3. long mutation fuzz against the UBSan binary, fresh random seed
#
# Exits non-zero if any step fails. Per-step logs go to build/premerge/.
# Knobs: PREMERGE_FUZZ_CASES (default 5000), PREMERGE_FUZZ_SEED (default random).
# UBSan only: ASan binaries hang at startup on the dev Mac.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/build/premerge"
FUZZ_CASES="${PREMERGE_FUZZ_CASES:-5000}"
FUZZ_SEED="${PREMERGE_FUZZ_SEED:-$RANDOM$RANDOM}"
UBSAN_FLAGS="-fsanitize=undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

mkdir -p "$LOGS"
names=()
results=()

# step <name> <command...>: run quietly into a log; show its tail on failure.
step() {
    local name="$1"; shift
    local log="$LOGS/$(echo "$name" | tr ' /' '__').log"
    printf '%-34s ' "$name"
    if "$@" >"$log" 2>&1; then
        echo "ok"
        results+=("ok")
    else
        echo "FAILED  (log: ${log#$ROOT/})"
        tail -25 "$log" | sed 's/^/    | /'
        results+=("FAILED")
    fi
    names+=("$name")
}

build() {  # build <dir> [cmake args...]
    local dir="$1"; shift
    cmake -S "$ROOT" -B "$ROOT/$dir" "$@" && cmake --build "$ROOT/$dir" -j "$JOBS"
}

run_ctest() { (cd "$ROOT/$1" && ctest --output-on-failure); }

echo "edgecc pre-merge gate  ($(git -C "$ROOT" rev-parse --abbrev-ref HEAD) @ $(git -C "$ROOT" rev-parse --short HEAD))"
step "build"                build build
step "ctest"                run_ctest build
step "build (UBSan)"        build build-ubsan -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_C_FLAGS=$UBSAN_FLAGS"
step "ctest (UBSan)"        run_ctest build-ubsan
step "fuzz $FUZZ_CASES cases (UBSan)" \
    python3 "$ROOT/tests/fuzz/run_fuzz.py" "$ROOT/build-ubsan/edgecc" \
        --cases "$FUZZ_CASES" --seed "$FUZZ_SEED"

failed=0
for r in "${results[@]}"; do [ "$r" = "ok" ] || failed=$((failed + 1)); done
echo
echo "fuzz seed: $FUZZ_SEED  (replay one case: tests/fuzz/run_fuzz.py build-ubsan/edgecc --seed $FUZZ_SEED --only <i>)"
if [ "$failed" -eq 0 ]; then
    echo "GATE PASSED (${#names[@]} steps)"
else
    echo "GATE FAILED ($failed of ${#names[@]} steps)"
    exit 1
fi
