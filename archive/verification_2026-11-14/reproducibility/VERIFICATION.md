# Verified offline artifact

Source commit: `12ff06c3287577a4c006ddaf348c177cc9515a7e` on `codex/reproducibility-artifact`.

Image: `mlirbench:aec` / `mlirbench:aec-verified`, `sha256:989cb18d05148998c4aec5711d8f89d0e9d03ed700a2720ac940d4186fe21039` (Linux AMD64).

The image was built exclusively from a committed Git archive, including a hash-verified 21-file data bundle. No source mounts, network, credentials or model calls were required for its runtime checks. Original `mlirbench:dev` remains available.

- **25 tests passed** in 47.622 seconds, including compiler integration, deterministic replay, input domains, judge safeguards, bundle integrity and recorded scoring.
- Both Linalg and Torch compiler/runtime smoke checks passed.
- Packaged master: **1,768 records**, original SHA-256 `84718efccae10e5bbd66fd7c4455ca9f22a8957c968151465b830855335a0541`.
- All **200** raw/processed working files retain their preservation hashes.
- The 26-pair, five-seed/two-repeat sample found **6 witnessed divergences, 10 no difference observed, 2 incompatible signatures, 5 compile errors, 3 unsupported Linalg preconditions**. Every one of its 160 successful program/input preflights remains valid under the final v3 policy. The sample is diagnostic/stratified, not an estimated dataset error rate.
- The final image reproduced the disputed Linalg CSV outputs using independent embedding; its inputs differ, so this is not a semantic counterexample.
- Invalid stored gather metadata was rejected as `INVALID_INPUT`. Explicit valid recorded-input replay succeeded in the preceding clean build; the final image's gather integration and replay tests also passed.
- Offline score aggregation exposes Gemini's 1,763 available evaluations versus 1,768 published pairs: 92.6471% hard-label accuracy over all pairs, 74.1806 mean stored judge score over available rows, or 73.9708 counting missing scores as zero.

No published labels, paper values, raw archives or Active reference source files were changed. New core source and the compressed data distribution are version-controlled. The bundle excludes only duplicate CSV distribution copies and non-dataset cache metadata; original working copies remain on disk.

## Remaining work

The artifact is **not submission-ready**. Resolve the three unsupported Linalg domain analyses, locate historical offline generation scripts and exact compiler provenance, expand replay coverage before full-dataset execution, verify native AMD64 behavior, freeze remaining apt/build dependencies, and locate/package the complete downstream GPU experiments and checkpoints. A separately corrected dataset or revised paper metrics would require an explicit publication decision; neither was created here.

See `ARTIFACT.md` for commands and `docs/REPRODUCIBILITY_AUDIT_2026-11-13.md` for detailed evidence, modifications, intermediate failures and limits. No changes were pushed remotely.
