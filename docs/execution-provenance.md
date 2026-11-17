# Execution provenance and current validation

The maintained validator uses positional shared inputs, operation-domain checks, deterministic candidate generation, two fresh-process repetitions and typed exact value comparison. Equal finite trials are reported as NO_DIFFERENCE_OBSERVED. They are not automatically minted into universal EQUIV labels. The pipeline adapter now respects the verdict of the complete suite instead of promoting a partial successful trial.

Input policy valid-domain-v5 attempts signed, positive, negative and zero-valued candidates, with deterministic domain-checked fallbacks. The first candidate pattern is selected by seed modulo four; each rejection and the accepted pattern are recorded. Gather indices are bounded for both programs. Logarithms, square roots, division and supported Torch operations are checked before execution. Unsupported syntax/domains and exhausted candidate searches stay unverified. Floating-point values are no longer forced positive for every program, which previously hid some sign-dependent differences.

IEEE-754 hexadecimal Torch float constants are supported, including infinity sentinels; unsupported literal/source syntax is distinguished from invalid input values. The original CSV does not store executed module bytes: program hashes bind the current published source to the frozen extraction sidecar. An additional independent check against the original mutation archive matched all 1,768 program pairs. Neither check reconstructs the actual historical runtime arguments or binary.

The small NumPy evaluator is used only to check input preconditions; IREE produces all audited outputs. Linalg helper/dynamic-domain analysis remains conservative. New checks refuse unparsed Torch operations, non-projection affine maps and additional unanalysed integer-conversion/domain operations.

## Original sources

The original Supabase execution_agent_unified_rows.csv is preserved byte-for-byte in data/historical/execution_agent_unified_rows.csv.gz. It has 1,804 rows and source SHA-256 08a14fa1f9f14ffdda5ec2e9191334113ebd44bcb3a73956eb15eb5bea6eff17. The audit checks its compression/source hashes, the extracted sidecar hash, all 1,768 published row hashes, program-source hashes and historical field values. Published master data and labels are unchanged.

This export does not record the actual per-program runtime inputs or the exact original toolchain identity. Its embedded-input fields are metadata, not proof of which values were executed. Input-domain flags therefore diagnose the metadata; they do not prove that invalid values were used in the historical execution.

Normal pipeline replay now enforces this frozen source verification before using historical execution observations. Resumed archived execution checkpoints are also checked against the original observations and program records, beyond their own checkpoint hashes. Quickstart data preparation, readiness checks, stored-metadata validation and historical-candidate replay use the same verification. Changed or missing sources stop the affected workflow. This checks record integrity without changing labels, score calculations, or the unresolved status of historical runtime-input provenance.

## Run

~~~bash
./ae.sh audit-provenance --output /results/provenance-run
./ae.sh audit-provenance --all --workers 4 --output /results/provenance-full
~~~

The default inventories all published records and runs a stratified diagnostic sample. The full command runs all pairs. Each uses five seeds and two fresh-process repeats per program/input. Use a fresh output directory. Quickstart demonstrates real execution separately from archived paper replay. The provenance audit is an additional author/reviewer diagnostic and is not run by default.

Files include source_verification.json, inventory.jsonl, canonical_audit.jsonl, linalg_historical_candidate.json and summary.json. Typed inputs and outputs are retained for independent replay. Workers operate on isolated cases; output order is deterministic.

## Interpret annotations

- DIVERGENCE_WITNESS_REPRODUCED: current repeated executions produced differing finite outputs for the same validated input.
- NO_COUNTEREXAMPLE_OBSERVED: tested finite outputs match; not a proof for all inputs.
- LABEL_COUNTEREXAMPLE: a valid new-input witness differs under typed exact comparison despite the published EQUIV label. Differences may include result dtypes or floating-point roundoff; inspect the lossless outputs and delta diagnostics before treating them as semantic label errors. This can challenge a general semantic-equivalence interpretation; it does not, by itself, disprove a historical observation on different inputs.
- STRUCTURAL_LABEL_CONFLICT: a published DIVERGE case has flat source text identical up to SSA value names and matching current valid-input trials. The structural detector preserves literals, quoted strings, symbols, types, operations and operand order, and refuses nested regions. It is not a general MLIR equivalence prover.
- PUBLISHED_WITNESS_NOT_REPRODUCED: a current valid-input witness was not found; this does not prove the label wrong.
- CURRENT_ERROR_OBSERVED: the current toolchain reports an error; this does not reconstruct the original error cause.
- UNVERIFIED: unsupported domains, incompatible signatures, failed generation, timeouts, nonfinite outputs or nondeterminism prevent a valid comparison.

The independent-embedding Linalg replay is deliberately separate. Reproducing recorded outputs using different inputs for A and B is not a semantic divergence witness and does not prove that this path generated the original label.

The execution_validation_system gate assesses the functioning audit system. The execution_label_provenance gate remains REVIEW_REQUIRED when historical runtime evidence is missing or scientific conflicts remain. Passing tests does not certify every published label. No numeric paper-score tolerance applies to these execution checks. A diagnostic sample is not a dataset error-rate estimate.
