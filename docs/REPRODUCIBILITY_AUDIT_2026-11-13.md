# Reproducibility investigation — November 13, 2026

The full handoff was read before edits. Work targets Camera-Ready/MLIRBench. The Active project was inspected read-only. Existing dirty/untracked work was retained.

## Verified baseline

HEAD is c39d788, main tracking origin/main. The shared-input helper and the pipeline/source/data copied earlier were untracked. The helper used original-derived embedding for both dialects and one sample; the old sample validator bypassed it entirely. `run.py check` was a placeholder. Only a package version test existed. Docker Desktop was stopped, then started for tests. The existing `mlirbench:dev` image was retained.

Master counts match the handoff: 1,768 = 888 Linalg + 880 Torch; string labels False 1,152, True 275, N/A 341. Actual status counts are success 1,400; expected_invalid 338; runtime_error 24; compile_error 3; missing 3. Labels and statuses are separate fields. Data preservation hashes cover 200 raw/processed files in `data/manifests/published_files_sha256.json`.

## Original execution evidence

The originally suggested JSON master has 1,768 flattened rows with null execution outputs and labels. Both execution_agent_unified.json and mlirbench_full_pipeline.json are empty lists. There is no `mlirbench_full_pipeline_rows2.csv` at the stated location. The useful file is `csv/execution_agent_unified_rows.csv` (1,804 records, SHA-256 `08a14fa1f9f14ffdda5ec2e9191334113ebd44bcb3a73956eb15eb5bea6eff17`). Every published ID has exactly one match. A new sidecar preserves the evidence without editing the export or published master.

For seed1_linalg_prog7_L1_R2, the original CSV records A maximum 799999.875 and B maximum 800001.0, an empty `pairs__input_spec_json`, and source files `pairs_benchmark_linalg_seed1.jsonl` / `judge_payloads_linalg_seed1_explained.jsonl`. These are the exact values predicted by independent embedding. Matching signatures do not ensure equal values when source constants drive separate embedding.

For torch_seed4_prog40_T1_R1, CSV outputs agree on softmax/argmax but differ in gathered loss values, including tiny apparent garbage values. Runtime input specs are [6,3] f32 and [6] si64. Stored embedded metadata includes index -1, outside Torch gather's nonnegative domain. Historical generic integer generation also permits out-of-bounds values. These outputs cannot establish a valid semantic counterexample without actual input provenance.

For torch_seed2_prog16_T2_R5, the mutation changes the input and most outputs from f32 to f64. The stored runtime spec is f32 only. Original A includes NaN strings; B contains extreme values. The new policy reports incompatible signatures and does not reinterpret storage or silently cast arguments. A historical buffer/type mismatch is a hypothesis, not a proven diagnosis.

The Active checked-in execution node and its Archive.zip copy both execute compile_and_run before generating embedded-input metadata. The historical source files named in the CSV were not found in the visible reference project. The archived Papers/diff.tex and diff2.tex are identical revision diff sources, not a verified final manuscript. Lines 653–680 explicitly describe embedded zero-argument Linalg and external Torch execution. This supports an offline generation pathway different from the checked-in node. It also asserts identical inputs for matching signatures, which the independent-embedding evidence contradicts. The sources state 74.18 aggregate / 92.65% hard-label accuracy for Gemini; those values have not been recomputed here.

## Implemented changes

- Strict signature parsing rejects dynamic or unsupported dimensions/types rather than dropping them; bf16 is no longer silently treated as float32.
- Lossless typed output transport preserves dtype/shape/bytes alongside the historical raw outputs. Empty tensor shapes survive subprocess input transport.
- Versioned domain generation and preflight cover bounded gather indices, supported Torch intermediate math, and conservative static Linalg cases. Unsupported preconditions produce explicit reports.
- A canonical suite compiles once per program, shares positional inputs, repeats executions, distinguishes errors/timeouts/nonfinite/nondeterminism, and records exact provenance. Equal trials do not become EQUIV.
- The pipeline adapter uses the suite, handles incompatible signatures without an exception, forwards actual trial inputs, skips benchmarks, and makes explanation generation opt-in.
- A separate historical embedding hypothesis replay uses original CSV outputs; stored metadata and explicitly recorded typed input replay remain distinct modes.
- Offline CLI commands, tests, reduced Docker context, explicit Docker copy paths, package installation and AE instructions were added. The older sample script delegates to canonical validation.

## Tests and outstanding findings

The initial targeted gather replay used indices [0,1,2,0,1,2] and standard-normal f32 data for seeds 42 and 123. Each program ran twice successfully. A/B outputs matched for both seeds. Exact inputs/outputs are in `reports/reproducibility/valid_gather_initial.json`.

Initial canonical tests passed the nine unit contracts and four compiler integration tests, including renamed arguments, changed arithmetic, replay, gather rejection and the execution node. The first run also exposed an unrelated package import failure in the old image; package installation and test source-path setup were added. Final test/build and sample outcomes are recorded below after completion.

Domain coverage is deliberately incomplete. Linalg helpers and runtime denominator/log/sqrt constraints need more analysis. Finite equal tests never prove equivalence. The complete historical label-generation code, original Torch-MLIR commit, native AMD64 verification, full offline score reproduction and downstream GPU experiments remain unresolved. No labels, paper values or historical source files were edited.


## Completed first sample and expanded tests

The first sample (policy valid-domain-v1) covered 26 unique records: 2 DIVERGE witnesses, 6 NO_DIFFERENCE_OBSERVED, 2 incompatible signatures, 5 compile errors, 11 unsupported preconditions. All five compile-error records carry published N/A, and only two were explicitly expected-invalid mutations. The reports preserve those distinctions. The published EQUIV seed1_linalg_prog10_L2_R1 changes its input signature and is incomparable under the new positional contract.

The initial historical candidate replay reproduced both original CSV outputs for seed1_linalg_prog7_L1_R2 on two runs per program; independently generated inputs differ. This is direct reproduction of the stored observation under the independent-embedding hypothesis, not proof of semantic divergence. It strengthens the explanation/provenance defect evidence without proving the exact original script path.

After adding Torch literal/device/shape operations and explicit invalid-domain tests, 17 unit/integration tests passed in the existing pinned image (50.817 seconds). The canonical judge now skips unverified results and initializes its model client lazily. The first sample and its three relevant source files are retained under reports/reproducibility/source_v1; the expanded policy is versioned valid-domain-v2.


The first rebuilt image (without source mounts, with `--network none`) passed all 17 tests in 61.343 seconds and retained the exact 1,768-record master hash. Its expanded v2 sample found 6 DIVERGE, 10 NO_DIFFERENCE_OBSERVED, 2 incompatible signatures, 5 compile errors, and 3 unsupported Linalg cases. All eight newly supported Torch strata/diagnostics completed; no invalid gather inputs were executed. The original 200-file raw/processed hash manifest was rechecked with zero changes. Every original/mutated source in the current master matches the original Supabase JSON master (zero differences across 3,536 programs).

The image's separate environment check exposed a Torch identity lowering problem: `input_type=auto` emitted an opaque `!torch.vtensor` runtime type rather than lowering it. Explicit `input_type=torch` compiled and executed the same function successfully. Compiler candidate order now prefers Torch explicitly and reports the selected input mode. A regression integration test was added; original v2 code and reports were retained before this change. Historical compiler artifacts and published values were not altered.


## Distribution and recorded-score findings (continued November 14)

The exact processed JSON/JSONL/README files total 501,966,581 bytes across 22 files. A deterministic gzip bundle is 20,712,724 bytes, with compressed/uncompressed hashes verified before unpacking. It avoids GitHub's per-file limit for the 201.9 MB reasoning JSONL and 188.2 MB errors JSONL. The original raw/processed working files remain physically unchanged; these working copies are ignored by Git in favor of the verified distribution bundle. Three bundle tests and two recorded-score tests were added. A transient test-file write error was caught by collection, corrected, and all seven host-compatible tests passed.

`score-recorded` exposes 15,907 evaluation rows (8 × 1,768 plus 1,763 Gemini). Five Gemini rows are absent, and the existing merge script explicitly verifies MODEL_FAILURE provenance for these IDs: seed3_linalg_prog24_L5_R4, seed3_linalg_prog24_L5_R5, seed3_linalg_prog25_L5_R5, seed3_linalg_prog30_L5_R4, seed3_linalg_prog30_L5_R5. All have published N/A / expected_invalid status. There are zero disagreements between recorded evaluation ground-truth labels and the published master across all nine models.

Recomputed Gemini hard-label accuracy is 92.6470588% over 1,768 pairs. The stored judge-score mean is 74.1806296 over 1,763 available rows; counting missing scores as zero gives 73.9708428 over 1,768 pairs. Thus the reported 92.65% and 74.18 use different denominators. These are exposed explicitly; no paper claims or historical labels were rewritten. Full subjective judge reproduction remains distinct from deterministic aggregation of stored scores.


The explicit-Torch-lowering image passed both offline compiler smoke checks and all 18 execution tests in 52.854 seconds. Repeating the complete 26-pair five-seed/two-repeat sample after this compiler-selection change produced the same categories: 6 witnessed DIVERGE, 10 NO_DIFFERENCE_OBSERVED, 2 incomparable signatures, 5 compile errors, 3 unsupported Linalg preconditions. See reports/reproducibility/final_sample.jsonl and its CSV/Markdown summaries. This targeted/stratified sample is not an estimate of the dataset's error rate.
