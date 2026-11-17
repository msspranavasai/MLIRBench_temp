# Disposition of the flagged execution cases

**30 cases documented; historical label provenance remains REVIEW_REQUIRED.**

This is a non-mutating review overlay. It preserves the published dataset, source programs, labels, saved responses, scoring rules, tolerances and paper results. The original audit assessments remain quoted as observations; the new disposition explains how to interpret them. No replacement ground-truth labels are adopted.

The review covers the 12 LABEL_COUNTEREXAMPLE cases, two STRUCTURAL_LABEL_CONFLICT cases, 15 PUBLISHED_WITNESS_NOT_REPRODUCED cases and one nondeterministic UNVERIFIED case in Syam’s complete audit. It does not resolve the separate unsupported-domain backlog.

## What was checked

- All 30 cases were replayed on the exact selected saved input bytes, with ten fresh-process repetitions per program. The batch produced 13 typed DIVERGE outcomes (including the selected nondeterministic case, which happened to look stable in ten runs) and 17 NO_DIFFERENCE_OBSERVED outcomes. These are current-input observations, not replacement labels.
- All 60 programs were accepted by the standalone verifier. The newer packaged IREE backend verifier accepted 58 and rejected the two type-changing L5 mutations. Only the core 2024 compiler was used for these fresh execution results: the newer tool supplied verifier evidence, not an independent execution backend.
- All five saved trials were additionally checked with a diagnostic per-output numpy.array_equal value comparison. This diagnostic does not change typed-exact comparison or introduce a tolerance.
- Stored metadata was validated before replay. Twenty metadata cases passed the current preflight and were replayed with ten repetitions (preflight acceptance does not establish source validity); ten Torch-family cases were rejected for an invalid gather index and were not executed with those inputs.
- The existing independent-embedding hypothesis was replayed for all 12 Linalg cases. It reproduced both stored outputs for 11 cases. Six of the seven unreproduced Linalg divergence witnesses were reproduced only with different original/mutated inputs; one remains unexplained. This is a historical-output hypothesis, not proof of the actual runtime arguments.
- Review execution used native ARM64 and the immutable rc4 image recorded in the JSON dossier. No paid APIs were called. It does not establish cross-architecture semantic behavior.

## Disposition summary

| Disposition | Cases |
|---|---:|
| NONDETERMINISTIC_ITERATOR_CONFLICT | 1 |
| INPUT_DEPENDENT_NUMERIC_DIFFERENCE | 2 |
| HISTORICAL_INPUT_HYPOTHESIS_SUPPORTED | 6 |
| INPUT_DEPENDENT_FP_DIFFERENCE | 4 |
| VERIFIER_VERSION_DEPENDENT_L5 | 2 |
| HISTORICAL_WITNESS_UNRESOLVED | 1 |
| MIXED_DTYPE_AND_PRECISION_CONTRACT | 4 |
| STRUCTURAL_CONFLICT_UNSAFE_METADATA | 2 |
| UNREPRODUCED_UNSAFE_METADATA | 8 |

## The nondeterministic case

`seed1_linalg_prog10_L2_R4` changes the output map from `(d0,d1)->(d0)` to `(d0,d1)->(d1)` but leaves the affected generic’s iterators `[parallel,reduction]`. Different parallel `d0` iterations then target the same `d1` output accumulator. This is an indexing/iteration conflict; syntactic verifier acceptance alone does not establish safe deterministic execution. The conflict is a source-level diagnosis supported by the following runtime control, not a recovered historical failure trace.

On saved seed 0, 100 fresh processes for each program produced:

| Program | Boolean vector | Observations |
|---|---|---:|
| Original | `[false,false,false]` | 100 |
| Published mutation | `[true,true,false]` | 96 |
| Published mutation | `[true,false,true]` | 4 |

A separate diagnostic copy changes only the affected iterator annotations to `[reduction,parallel]`. Twenty repetitions were stable: the original yielded `[false,false,false]`, and the diagnostic copy yielded `[true,false,false]`, matching independent row-wise and column-wise NumPy AND calculations on the same mask. This copy is not written into the dataset or used for paper scoring. Its stability supports the diagnosis; it is not a proof that every compiler issue is excluded.

**Disposition:** quarantine this pair from new claims of a stable valid-input semantic witness. Preserve the published DIVERGE record as a historical observation. A new batch need not reproduce the same 96/4 frequency, or observe variation within a short repeat budget.

## Comparison-contract differences

The manuscript names numpy.array_equal. Both comparisons require matching shapes; the current typed validator additionally requires matching output dtypes. Four selected witnesses in the old 12-case bucket have equal numeric values but different dtypes. Looking across all five seeds is essential: three of these pairs also have real numeric precision differences on seed 123. Only `torch_seed3_prog28_T2_R5` is numerically equal on all five saved generated trials while typed-exact still reports divergence. An additional replay of preflight-accepted stored metadata exposes a numeric precision difference for this pair too. Thus all four dtype-changing pairs have a numeric witness in the expanded review; this does not prove those metadata values were used historically.

Therefore these flags are not all interchangeable evidence of incorrect historical labels. The dossier records both contracts separately. The current validator is left unchanged, and no allclose/epsilon rule is introduced. See the [NumPy comparison definition](https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html).

## Unsafe historical metadata and structural conflicts

The ten reviewed `torch_seed4_prog40_*` cases share stored gather indices `[0,1,1,1,2,-1]` for a dimension of size 3. The negative index is invalid for gather; the current preflight rejects it. The historical loss outputs contain tiny unexpected values in this family, but this review does not claim that the stored metadata was actually executed or that an out-of-bounds read has been proven as the original cause.

Two pairs, T1_R1 and T1_R4, are identical after flat SSA-name normalization and agree under the valid shared-input trials. Their structural conflicts are retained and documented; the archived DIVERGE values are not silently changed. The other eight family pairs also agree on the tested valid inputs, but finite agreement is not universal equivalence.

See the [PyTorch CPU gather bounds check](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/ScatterGatherKernel.cpp) and [MLIR SSA language reference](https://mlir.llvm.org/docs/LangRef/).

## Compiler-version-dependent L5 cases

`seed1_linalg_prog9_L5_R2` changes an i1 fill seed to f32; `seed4_linalg_prog33_L5_R2` changes an f32 fill seed to i64. The older verifier/compiler accepts these sources. The packaged IREE 3.12.0rc20260916 verifier rejects their mismatched fill value/output element types. The legacy execution differences are real observations, but a single cross-version validity or corrected label is not justified. Keep the compiler identities and diagnostics with these cases.

## Case-by-case decisions

### seed1_linalg_prog10_L2_R4

Published label: **DIVERGE**. Original audit assessment: `UNVERIFIED`.

**Disposition: NONDETERMINISTIC_ITERATOR_CONFLICT.** Output map changes d0 to d1, but iterator types remain [parallel,reduction]. Different d0 iterations target the same d1 output accumulator. The saved source is accepted by both verifiers; repeated execution is not stable. A separate iterator-only diagnostic repair is stable and matches a column-wise NumPy reference.

**Action:** Quarantine from new deterministic semantic-label claims; preserve the original program and recorded DIVERGE label.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `DIVERGE`. Independent-embedding hypothesis: `HISTORICAL_OUTPUTS_REPRODUCED`.

### seed1_linalg_prog4_L3_R4

Published label: **EQUIV**. Original audit assessment: `LABEL_COUNTEREXAMPLE`.

**Disposition: INPUT_DEPENDENT_NUMERIC_DIFFERENCE.** The divisor changes from 9 to 1. Independently generated shared arguments can differ, exposing a nonzero mean absolute difference multiplied by nine. Stored metadata has equal arguments and yields zero in both versions.

**Action:** Retain the current-suite counterexample and the archived EQUIV observation as different input scopes; do not claim recovery of actual historical inputs.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `HISTORICAL_OUTPUTS_REPRODUCED`.

### seed1_linalg_prog5_L2_R3

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: HISTORICAL_INPUT_HYPOTHESIS_SUPPORTED.** The only source edit changes the NaN-count divisor from 12 to 1. With finite inputs, cmpf une(x,x) is false, so the numerator is zero and this output is unchanged. All shared-input trials agree. Independently embedded inputs differ and reproduce both stored outputs, including their differing final sums; this is evidence for a different-input historical hypothesis, not a semantic divergence.

**Action:** Preserve the original observation; do not use different-input output replay as a semantic divergence witness. Do not assert universal equivalence.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `HISTORICAL_OUTPUTS_REPRODUCED`.

### seed1_linalg_prog5_L4_R5

Published label: **EQUIV**. Original audit assessment: `LABEL_COUNTEREXAMPLE`.

**Disposition: INPUT_DEPENDENT_FP_DIFFERENCE.** The mutation fuses compare and cast into a multi-output linalg.generic. The signed seed-0 input exposes an exact floating-point difference in the returned sum; stored metadata replay agrees. The precise compiler pass causing the low-bit difference has not been isolated.

**Action:** Retain as an exact current-suite difference, with compiler-layout sensitivity unresolved; do not add an output tolerance.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `HISTORICAL_OUTPUTS_REPRODUCED`.

### seed1_linalg_prog7_L1_R2

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: HISTORICAL_INPUT_HYPOTHESIS_SUPPORTED.** All five saved shared-input trials agree. Ten-repeat replay agrees. Independently embedded original/mutated inputs differ and reproduce both stored historical outputs. This explains those outputs under a concrete hypothesis, not a recovered actual-runtime trace.

**Action:** Preserve the original observation; do not use different-input output replay as a semantic divergence witness. Do not assert universal equivalence.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `HISTORICAL_OUTPUTS_REPRODUCED`.

### seed1_linalg_prog7_L4_R3

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: HISTORICAL_INPUT_HYPOTHESIS_SUPPORTED.** All five saved shared-input trials agree. Ten-repeat replay agrees. Independently embedded original/mutated inputs differ and reproduce both stored historical outputs. This explains those outputs under a concrete hypothesis, not a recovered actual-runtime trace.

**Action:** Preserve the original observation; do not use different-input output replay as a semantic divergence witness. Do not assert universal equivalence.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `HISTORICAL_OUTPUTS_REPRODUCED`.

### seed1_linalg_prog9_L5_R2

Published label: **EQUIV**. Original audit assessment: `LABEL_COUNTEREXAMPLE`.

**Disposition: VERIFIER_VERSION_DEPENDENT_L5.** The fill seed changes from i1 true to f32 zero for an i1 output. LLVM19 and IREE 2024 accept the source; the packaged IREE 3.12 backend verifier rejects the mismatched fill type. Legacy execution also changes the boolean reduction on saved seed 123.

**Action:** Treat as a version-dependent verification/error-test case, not an unconditional well-defined semantic counterexample. Preserve the archived label.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `HISTORICAL_OUTPUTS_REPRODUCED`.

### seed2_linalg_prog15_L1_R2

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: HISTORICAL_WITNESS_UNRESOLVED.** Operation scheduling changes are present. Shared-input saved trials and ten-repeat replay agree, but independent-embedding replay does not reproduce either stored historical output.

**Action:** Historical divergence remains unresolved. Matching finite trials neither recovers the old inputs nor proves universal equivalence.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `HISTORICAL_OUTPUTS_NOT_REPRODUCED`.

### seed3_linalg_prog21_L2_R5

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: HISTORICAL_INPUT_HYPOTHESIS_SUPPORTED.** All five saved shared-input trials agree. Ten-repeat replay agrees. Independently embedded original/mutated inputs differ and reproduce both stored historical outputs. This explains those outputs under a concrete hypothesis, not a recovered actual-runtime trace.

**Action:** Preserve the original observation; do not use different-input output replay as a semantic divergence witness. Do not assert universal equivalence.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `HISTORICAL_OUTPUTS_REPRODUCED`.

### seed3_linalg_prog22_L1_R2

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: HISTORICAL_INPUT_HYPOTHESIS_SUPPORTED.** All five saved shared-input trials agree. Ten-repeat replay agrees. Independently embedded original/mutated inputs differ and reproduce both stored historical outputs. This explains those outputs under a concrete hypothesis, not a recovered actual-runtime trace.

**Action:** Preserve the original observation; do not use different-input output replay as a semantic divergence witness. Do not assert universal equivalence.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `HISTORICAL_OUTPUTS_REPRODUCED`.

### seed4_linalg_prog33_L1_R2

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: HISTORICAL_INPUT_HYPOTHESIS_SUPPORTED.** All five saved shared-input trials agree. Ten-repeat replay agrees. Independently embedded original/mutated inputs differ and reproduce both stored historical outputs. This explains those outputs under a concrete hypothesis, not a recovered actual-runtime trace.

**Action:** Preserve the original observation; do not use different-input output replay as a semantic divergence witness. Do not assert universal equivalence.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `HISTORICAL_OUTPUTS_REPRODUCED`.

### seed4_linalg_prog33_L5_R2

Published label: **EQUIV**. Original audit assessment: `LABEL_COUNTEREXAMPLE`.

**Disposition: VERIFIER_VERSION_DEPENDENT_L5.** The row-maximum initializer changes from a floating sentinel to i64 zero for an f32 tensor. The newer backend verifier rejects that fill type; the legacy compiler accepts it. Negative seed-42 inputs expose zeros instead of negative row maxima and changed argmax indices.

**Action:** Retain toolchain-specific acceptance and execution evidence; do not assign one cross-version corrected label.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `HISTORICAL_OUTPUTS_REPRODUCED`.

### torch_seed1_prog3_T2_R3

Published label: **EQUIV**. Original audit assessment: `LABEL_COUNTEREXAMPLE`.

**Disposition: INPUT_DEPENDENT_NUMERIC_DIFFERENCE.** The equality-count divisor changes from 9 to 8. With all-one equal arguments (saved seed 123), the fraction changes from 1 to 1.125. Archived outputs have zero equality count, where either divisor gives zero. Safe metadata replay differs, showing that metadata is not sufficient to prove the historical runtime.

**Action:** Document a genuine current-input numeric difference; do not silently overwrite the fixed historical label or claim the metadata was executed.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `DIVERGE`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed1_prog3_T2_R5

Published label: **EQUIV**. Original audit assessment: `LABEL_COUNTEREXAMPLE`.

**Disposition: MIXED_DTYPE_AND_PRECISION_CONTRACT.** The mutation changes returned scalar f32 computations to f64. Four saved trials are numerically equal but dtype-different; seed 123 also has a genuine numeric precision difference.

**Action:** Separate dtype-only flags from numeric divergence. Do not interpret every typed-exact failure as a paper-style value-comparison failure.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `DIVERGE`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed1_prog4_T2_R5

Published label: **EQUIV**. Original audit assessment: `LABEL_COUNTEREXAMPLE`.

**Disposition: MIXED_DTYPE_AND_PRECISION_CONTRACT.** The mutation changes returned f32 scalar computations to f64. Four saved trials differ only in dtype; seed 123 also differs numerically.

**Action:** Retain both comparison results and preserve the original label; no tolerance or dtype-coercion change is applied to the validator.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `DIVERGE`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed2_prog19_T3_R2

Published label: **EQUIV**. Original audit assessment: `LABEL_COUNTEREXAMPLE`.

**Disposition: INPUT_DEPENDENT_FP_DIFFERENCE.** A vector matmul/dot is rewritten as multiply followed by sum. Seed 42 exposes a low-bit result difference; safe metadata replay agrees. Arithmetic grouping/lowering changes are implicated, not a proven defect in a particular compiler pass.

**Action:** Retain the exact current-suite difference and document input/precision sensitivity; original runtime provenance remains unproven.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed3_prog28_T2_R5

Published label: **EQUIV**. Original audit assessment: `LABEL_COUNTEREXAMPLE`.

**Disposition: MIXED_DTYPE_AND_PRECISION_CONTRACT.** The returned row output changes from f32 to f64. All five saved generated trials are numerically equal and differ only in dtype. Expanding the review to preflight-accepted stored metadata also exposes a real numeric precision difference. That metadata is not proven to be the actual historical input.

**Action:** Separate the dtype-only saved trials from the additional numeric witness; do not dismiss the pair or silently adopt a corrected historical label.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `DIVERGE`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog35_T3_R2

Published label: **EQUIV**. Original audit assessment: `LABEL_COUNTEREXAMPLE`.

**Disposition: INPUT_DEPENDENT_FP_DIFFERENCE.** Variance calculations using f64 accumulation are rewritten to f32 paths. Seeds 1 and 2026 expose low-bit differences; safe metadata replay agrees.

**Action:** Document precision-dependent behavior rather than rounding it away or treating mathematical equivalence as machine equality.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog36_T2_R5

Published label: **EQUIV**. Original audit assessment: `LABEL_COUNTEREXAMPLE`.

**Disposition: MIXED_DTYPE_AND_PRECISION_CONTRACT.** The fraction/one-minus-fraction output changes from f32 to f64. Four saved trials are numerically equal; seed 123 also differs numerically.

**Action:** Report contract and precision effects separately; preserve historical scoring.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `DIVERGE`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog38_T3_R2

Published label: **EQUIV**. Original audit assessment: `LABEL_COUNTEREXAMPLE`.

**Disposition: INPUT_DEPENDENT_FP_DIFFERENCE.** pow(x,2) is rewritten as x*x in a variance computation. Seeds 42 and 123 expose low-bit output differences; safe metadata replay agrees. The lowering-level cause is not fully isolated.

**Action:** Retain exact differences and the limitation on backend-independent conclusions.

Fresh saved-witness replay: `DIVERGE`; ten repeats per program. Stored-metadata result: `NO_DIFFERENCE_OBSERVED`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog40_T1_R1

Published label: **DIVERGE**. Original audit assessment: `STRUCTURAL_LABEL_CONFLICT`.

**Disposition: STRUCTURAL_CONFLICT_UNSAFE_METADATA.** The pair is token-identical after flat SSA-name normalization. This family metadata contains gather index -1 for a dimension of size 3. Preflight rejects it; this review does not execute those unsafe metadata values. Metadata is not proof of actual historical input use.

**Action:** Retain the historical record for paper replay, but quarantine the historical divergence from claims of a validated valid-input semantic witness. Actual historical cause remains unproven.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `INVALID_INPUT`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog40_T1_R2

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: UNREPRODUCED_UNSAFE_METADATA.** The mutation is a constant reorder, copy flag, common-subexpression reuse, or value-preserving structural rewrite; current tested executions agree. This family metadata contains gather index -1 for a dimension of size 3. Preflight rejects it; this review does not execute those unsafe metadata values. Metadata is not proof of actual historical input use.

**Action:** Retain the historical record for paper replay, but quarantine the historical divergence from claims of a validated valid-input semantic witness. Actual historical cause remains unproven.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `INVALID_INPUT`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog40_T1_R4

Published label: **DIVERGE**. Original audit assessment: `STRUCTURAL_LABEL_CONFLICT`.

**Disposition: STRUCTURAL_CONFLICT_UNSAFE_METADATA.** The pair is token-identical after flat SSA-name normalization. This family metadata contains gather index -1 for a dimension of size 3. Preflight rejects it; this review does not execute those unsafe metadata values. Metadata is not proof of actual historical input use.

**Action:** Retain the historical record for paper replay, but quarantine the historical divergence from claims of a validated valid-input semantic witness. Actual historical cause remains unproven.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `INVALID_INPUT`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog40_T2_R5

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: UNREPRODUCED_UNSAFE_METADATA.** The mutation is a constant reorder, copy flag, common-subexpression reuse, or value-preserving structural rewrite; current tested executions agree. This family metadata contains gather index -1 for a dimension of size 3. Preflight rejects it; this review does not execute those unsafe metadata values. Metadata is not proof of actual historical input use.

**Action:** Retain the historical record for paper replay, but quarantine the historical divergence from claims of a validated valid-input semantic witness. Actual historical cause remains unproven.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `INVALID_INPUT`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog40_T3_R5

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: UNREPRODUCED_UNSAFE_METADATA.** The mutation is a constant reorder, copy flag, common-subexpression reuse, or value-preserving structural rewrite; current tested executions agree. This family metadata contains gather index -1 for a dimension of size 3. Preflight rejects it; this review does not execute those unsafe metadata values. Metadata is not proof of actual historical input use.

**Action:** Retain the historical record for paper replay, but quarantine the historical divergence from claims of a validated valid-input semantic witness. Actual historical cause remains unproven.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `INVALID_INPUT`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog40_T4_R1

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: UNREPRODUCED_UNSAFE_METADATA.** The mutation is a constant reorder, copy flag, common-subexpression reuse, or value-preserving structural rewrite; current tested executions agree. This family metadata contains gather index -1 for a dimension of size 3. Preflight rejects it; this review does not execute those unsafe metadata values. Metadata is not proof of actual historical input use.

**Action:** Retain the historical record for paper replay, but quarantine the historical divergence from claims of a validated valid-input semantic witness. Actual historical cause remains unproven.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `INVALID_INPUT`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog40_T4_R2

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: UNREPRODUCED_UNSAFE_METADATA.** The mutation is a constant reorder, copy flag, common-subexpression reuse, or value-preserving structural rewrite; current tested executions agree. This family metadata contains gather index -1 for a dimension of size 3. Preflight rejects it; this review does not execute those unsafe metadata values. Metadata is not proof of actual historical input use.

**Action:** Retain the historical record for paper replay, but quarantine the historical divergence from claims of a validated valid-input semantic witness. Actual historical cause remains unproven.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `INVALID_INPUT`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog40_T4_R3

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: UNREPRODUCED_UNSAFE_METADATA.** The mutation is a constant reorder, copy flag, common-subexpression reuse, or value-preserving structural rewrite; current tested executions agree. This family metadata contains gather index -1 for a dimension of size 3. Preflight rejects it; this review does not execute those unsafe metadata values. Metadata is not proof of actual historical input use.

**Action:** Retain the historical record for paper replay, but quarantine the historical divergence from claims of a validated valid-input semantic witness. Actual historical cause remains unproven.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `INVALID_INPUT`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog40_T4_R4

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: UNREPRODUCED_UNSAFE_METADATA.** The mutation is a constant reorder, copy flag, common-subexpression reuse, or value-preserving structural rewrite; current tested executions agree. This family metadata contains gather index -1 for a dimension of size 3. Preflight rejects it; this review does not execute those unsafe metadata values. Metadata is not proof of actual historical input use.

**Action:** Retain the historical record for paper replay, but quarantine the historical divergence from claims of a validated valid-input semantic witness. Actual historical cause remains unproven.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `INVALID_INPUT`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

### torch_seed4_prog40_T4_R5

Published label: **DIVERGE**. Original audit assessment: `PUBLISHED_WITNESS_NOT_REPRODUCED`.

**Disposition: UNREPRODUCED_UNSAFE_METADATA.** The mutation is a constant reorder, copy flag, common-subexpression reuse, or value-preserving structural rewrite; current tested executions agree. This family metadata contains gather index -1 for a dimension of size 3. Preflight rejects it; this review does not execute those unsafe metadata values. Metadata is not proof of actual historical input use.

**Action:** Retain the historical record for paper replay, but quarantine the historical divergence from claims of a validated valid-input semantic witness. Actual historical cause remains unproven.

Fresh saved-witness replay: `NO_DIFFERENCE_OBSERVED`; ten repeats per program. Stored-metadata result: `INVALID_INPUT`. Independent-embedding hypothesis: `NOT_RUN_TORCH_UNSAFE_HISTORICAL_PATH`.

## Reproduce the review without model APIs

The machine-readable [dossier](label-review/dispositions.json) includes exact saved witness inputs, program hashes, comparison observations and raw-review file hashes. [The integrity manifest](../experiments/label_review_manifest.json) pins it to the immutable published master. Raw diagnostic runs remain outside Git; the witness command regenerates new results. Existing results are never overwritten.

```bash
# Verify the 30-case dossier and its links to the published snapshot.
./ae.sh review-labels --output /results/label-review

# Replay all selected saved witnesses with ten repetitions per program.
./ae.sh review-labels --fresh --all --repeats 10 --output /results/label-review-fresh

# Probe the nondeterministic case with a longer fresh-process run.
./ae.sh review-labels --fresh --id seed1_linalg_prog10_L2_R4 --repeats 100 \
  --output /results/nondeterminism-review

# Compare a clearly separated diagnostic iterator correction.
./ae.sh review-labels --fresh --id seed1_linalg_prog10_L2_R4 --iterator-control \
  --repeats 20 --output /results/iterator-control

# For a mixed dtype/precision case, use its additional numeric witness.
./ae.sh review-labels --fresh --id torch_seed1_prog3_T2_R5 --numeric-witness \
  --output /results/numeric-witness

# Revisit a documented historical Linalg embedding hypothesis.
./ae.sh historical-candidate --id seed1_linalg_prog5_L2_R3 \
  --output /results/embedding-hypothesis.json
```

The fresh review defaults to one case unless --all or explicit IDs are supplied. All four dtype/precision cases have an additional numeric witness: three from saved seed 123 and one from preflight-accepted stored metadata. Fresh replay does not automatically overwrite a documented disposition or assign new labels. `DOCUMENTED_WITH_LIMITATIONS` means the review record is complete for the selected scope, not that label provenance is proven.

## What this does and does not settle

Every selected case now has a disposition and an evidence-backed next action. Six numeric differences remain current-input observations; four dtype-changing pairs have comparison-contract effects on selected saved trials and also precision-sensitive numeric witnesses in the expanded review; two L5 cases depend on verifier version; the nondeterministic case is quarantined as a source/iteration risk; historical witnesses are either supported by a different-input hypothesis, associated with unsafe metadata, or explicitly unresolved.

The published fixed-input observations and the new shared-input tests have different scopes. Even a genuine new counterexample does not recover the missing original runtime arguments. We do not fabricate a corrected-label table, claim every old label is wrong, or certify unchanged scientific conclusions under a hypothetical relabeling. Frozen paper calculations remain reproducible and unchanged. The global historical-label gate remains REVIEW_REQUIRED; the wider unsupported-domain backlog and GPU/other release limitations remain separate.

Technical context: [Linalg semantics](https://mlir.llvm.org/docs/Dialects/Linalg/) and [iterator inference from output maps](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Linalg/IR/LinalgInterfaces.cpp).
