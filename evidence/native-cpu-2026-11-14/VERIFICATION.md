# Native CPU verification

The Dockerfile now supports Linux ARM64 and AMD64 with identical dependency versions. The launcher defaults to the Docker daemon's architecture, uses separate architecture tags, and records emulation status. Explicit AE_PLATFORM overrides remain available.

- Native ARM64: 35 tests passed without skips; both compiler smoke checks passed; all 76 downstream comparisons matched. Offline quickstart took 11.441 seconds. Docker exposed 16 logical/affinity CPUs and reported no instruction emulation.
- Explicit AMD64 on this ARM64 host: 35 tests passed without skips; compiler smoke checks and all 76 downstream comparisons passed. Its report correctly identifies emulation. This does not establish native AMD64 hardware performance.
- Compiler versions, shared-input rules and numeric comparison policy are unchanged. CPU timing here is workflow duration, not a replacement for paper benchmark measurements.
- Two published diagnostic pairs executed with seeds 42 and 123 on ARM64, with two fresh-process repeats per program/input. Both yielded NO_DIFFERENCE_OBSERVED, consistent with prior audits; the historical DIVERGE labels remain unreproduced and unchanged.
- Published data verified in both runs. Existing author edits to the level-table calculation and paper-reference selectors were preserved byte-for-byte; their hashes are recorded separately because the working tree includes these edits.
- Source fingerprints now include the Dockerfile, launcher and pinned core requirements.

The native_cpu_execution readiness gate passes for the ARM64 run. Native AMD64 evaluator verification and other submission requirements remain open. The artifact is still NOT_SUBMISSION_READY.

Use ./ae.sh quickstart. The local image is mlirbench-ae:dev-arm64; older AMD64 images are preserved and may still display Docker Desktop's emulation warning.
