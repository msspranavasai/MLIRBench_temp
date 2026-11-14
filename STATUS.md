# Start here: AE preparation status

**Organized for AE preparation; not submission-ready yet.**

Default evaluation uses archived model responses/judge observations, downstream saved predictions and actual CPU compilation/execution.

Established in this preparation work:

- Published input bundles are preserved and checksum-verified.
- Downstream source, predictions and original splits were located on Nova and copied read-only.
- All 76 downstream metric comparisons match archived final-figure CSVs.
- All 13 family-split checks passed; DT1 and DT2 prediction IDs/gold labels match the original held-out files.
- The organized package passed 31 core tests and actual compiler execution in both workspace and clean-source builds.
- Provider code and dependencies are optional.

Remaining author work:

1. Reconcile level-wise aggregation against the confirmed accepted paper; preserve current numerical differences for review.
2. Reconcile historical input/label provenance and state supported execution claims accurately.
3. Validate portable GPU/checkpoint evaluation and arrange distribution/access.
4. Complete native AMD64 and final dependency/resource verification. The launcher now selects native ARM64 on Apple Silicon; the native_cpu_execution gate separately reports whether the current run uses emulation.

Run ./ae.sh quickstart and read the generated results/<run>/START_HERE.md. It reports actual checks and open gates. A passing test suite does not substitute for those remaining requirements.

The checked-in baseline is evidence/preparation-2026-11-14/VERIFICATION.md. Full quickstart runtime was 39.055 seconds on the local emulated CPU setup.


Native CPU update: 35 tests and all 76 downstream comparisons passed on ARM64 without emulation; offline quickstart took 12.867 seconds. Explicit AMD64 emulation also passed. See evidence/native-cpu-2026-11-14-final/VERIFICATION.md. Native AMD64 hardware remains unverified.
