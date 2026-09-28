# Pre-merge gate Implementation Plan

**Goal:** One local command (`tools/premerge.sh`) that runs the normal suite,
a UBSan suite, and a long mutation fuzz; plus two new ctest suites (fuzz,
independent reference) and CLAUDE.md process docs.

**Spec:** `docs/superpowers/specs/2026-09-28-edgecc-premerge-gate-design.md`

- [ ] **Task 1: Mutation fuzzer.** `tests/fuzz/run_fuzz.py <edgecc> [--cases N] [--seed S]`;
  wire as ctest `fuzz_tests` (fixed seed, runs in seconds). Verify it catches
  the known bug class: run it against a binary built from `763024b` (pre-fix)
  and see failures; against HEAD see none. Triage anything new it finds:
  small bugs get a failing test then a fix; bigger ones are recorded, not fixed here.
- [ ] **Task 2: Reference suite.** Kernels in `tests/reference/kernels/`,
  harness `tests/reference/run_reference.py <edgecc>`; wire as ctest
  `reference_tests`; scipy-optional skip. Sanity-check the oracle by
  confirming a deliberately wrong kernel fails.
- [ ] **Task 3: `tools/premerge.sh`** + `.gitignore` `build-ubsan/`.
  Run it end to end.
- [ ] **Task 4: Docs.** CLAUDE.md: Test section (new suites, gate) and a new
  "Adding a feature" process section; roadmap row reconciliation.
- [ ] **Task 5: Finish.** Run the gate; reconcile spec deviations; tick these
  boxes; merge `--no-ff`; update memory.
