# MLIRBench CGO 2027 Artifact Evaluation: Complete Codex Handoff

**Prepared:** November 13, 2026
**Audience:** Codex running locally with access to the NEW Camera-Ready MLIRBench Git repository
**Task:** Continue implementation and reproducibility work. Inspect the actual filesystem before changing anything. Treat the old `Active/` project as a reference, not the destination for edits.

## 0. Your role and operating instructions

You are taking over work to prepare the artifact for the accepted **CGO 2027 Practical/Empirical paper, “MLIRBench: A Benchmark and Dataset for Compiler and Hardware-Aware Reasoning” (submission #309)**. Make the single-container artifact accurate, reproducible, self-contained, and straightforward for artifact evaluators to run. Work in the **NEW** Git repository; use the older `Active/` project, GitHub repository, and original logs as historical evidence.

- Start by inspecting `pwd`, `git status`, repository files, Dockerfile, requirements, tests, and the code described here. This handoff is detailed historical context, not a substitute for inspecting actual code. Identify anything that has diverged.
- **Never overwrite or silently rewrite the published dataset, paper results, or original archives.** All verification outputs should go to separate reports with provenance.
- Never claim that a failed-to-reproduce DIVERGE label is proven EQUIV. Finite testing can find a counterexample but cannot prove full semantic equivalence.
- Do not “fix” labels by comparing independently generated inputs, forcing the expected label, introducing numeric tolerances, or treating invalid runtime inputs as valid divergence.
- Do not commit secrets, `.env` files containing credentials, cloud service tokens, or API keys. Model-provider API calls have potential costs and should be optional/opt-in for AEC.
- Avoid modifying the original `Active/` directory. Copy only selectively and with clear attribution/provenance. Preserve tracked files and existing uncommitted work. No destructive git operations.
- Prefer tests and reproducible command-line entry points over complicated interactive flows. A single Docker image is desired; no Docker Compose, remote Torch-MLIR validator, or GCP dependency for core verification.
- The user has preferred one clear actionable terminal command at a time during interactive troubleshooting. As Codex, you have local folder access, so implement and test directly, then summarize what you changed and what remains.

## 1. Local directories: new artifact versus old `Active/` project

### NEW repository: primary writable workspace

```text
~/Documents/ISU/PhD_Iowa_State/03_Projects/Camera-Ready/MLIRBench
```

Equivalent to `/Users/pranava/Documents/ISU/PhD_Iowa_State/03_Projects/Camera-Ready/MLIRBench` on the user's Mac. This is the new Git repository and the directory to modify for CGO AEC.

Recent Git state seen in the user's terminal:

```text
branch: main
HEAD: c39d788  build: configure Python 3.11 development environment
      c29b57a  chore: establish CGO 2027 artifact structure
      415103d  Initial commit
origin/main: c39d788
```

The shell prompt displayed `main !?`, indicating dirty/untracked work. In particular, `agents/execution_sandbox_agent/node.py` was **untracked** (`??`) when tested, so `git log --follow` gave no history. Check and commit all necessary source and tests as part of packaging, but only after reviewing changes.

### OLD `Active/` workspace: historical/reference project, do not modify

**Active root:**

```text
/Users/pranava/Documents/ISU/PhD_Iowa_State/03_Projects/Active/MLIRBench
```

**Original full pipeline source:**

```text
/Users/pranava/Documents/ISU/PhD_Iowa_State/03_Projects/Active/MLIRBench/Code/MLIRBench
```

Contains `agents/`, `graph/`, `services/`, `run.py`, `requirements.txt`, and `docs/`. The historical `.mlirbench` Python virtual environment is at:

```text
/Users/pranava/Documents/ISU/PhD_Iowa_State/03_Projects/Active/MLIRBench/Code/MLIRBench/.mlirbench
```

**Original data roots:**

```text
/Users/pranava/Documents/ISU/PhD_Iowa_State/03_Projects/Active/MLIRBench/Data/External
/Users/pranava/Documents/ISU/PhD_Iowa_State/03_Projects/Active/MLIRBench/Data/Processed
/Users/pranava/Documents/ISU/PhD_Iowa_State/03_Projects/Active/MLIRBench/Data/Raw
/Users/pranava/Documents/ISU/PhD_Iowa_State/03_Projects/Active/MLIRBench/Data/Supabase
/Users/pranava/Documents/ISU/PhD_Iowa_State/03_Projects/Active/MLIRBench/Data/Supabase/supabase_export
```

Known export subdirectories/files (verify their continued existence):

```text
Active/MLIRBench/Data/Supabase/supabase_export/csv/
Active/MLIRBench/Data/Supabase/supabase_export/json/mlirbench_master.json
Active/MLIRBench/Data/Supabase/supabase_export/csv/mlirbench_core_hf.csv
Active/MLIRBench/Data/Supabase/supabase_export/csv/mlirbench_full_pipeline_rows2.csv
Active/MLIRBench/Data/Supabase/supabase_export/export_summary.json
```

The CSV placement/filename and `export_summary.json` placement should be checked on disk, not assumed; these were previously referenced in prior commands and documents. The full JSON export path was explicitly used historically. These exports may contain raw/nested execution output fields missing from the flattened new master JSONL. **Investigate the original `programA_output` and `programB_output` for disputed records here before making dataset corrections.**

Other folders previously listed under the original `Active/MLIRBench` root:

```text
Admin/
Code/
Conferences/
Data/
Downstream Tasks/
Experiments/
Human Grading/
Ideas/
Literature/
Meetings/
Papers/
Presentations/
Results/
```

The `Papers/`, `Downstream Tasks/`, `Results/`, `Experiments/`, and `Human Grading/` folders are potentially important for paper claims, original downstream-task code, outputs, and analysis. Inspect them directly. Do not assume filenames in them.

**Other historical compiler environment, outside `Active/`:**

```text
/Users/pranava/Documents/ISU/S26/ug_fs_2/mascod-bench/mascod_env
/Users/pranava/Documents/ISU/S26/ug_fs_2/mascod-bench/mascod_env/bin/iree-opt
```

These paths were used in older Spring 2026 research work. They are not automatically the canonical CGO artifact toolchain, but may provide useful historical compiler-version evidence. Check existence and versions, if needed.

### Historical GitHub repository

```text
https://github.com/msspranavasai/MLIRBench_old.git
```

Public repository, default `main`, final reviewed commit `5f328ad081390b5d189eb94aec09fa0a4db4336d` dated June 12, 2026. Important commits:

- `4d6bf44549e3108b0fbb34b728a2e7df36e22957` (June 6, 2026): execution tool and agent implementation recorded.
- `2a34f934276038bc1e71bf59a339f16cf7e739d9` (June 7, 2026): pipeline/cloud validation changes.
- `5f328ad081390b5d189eb94aec09fa0a4db4336d` (June 12, 2026): later files/website.

Useful files in historical repo:

```text
agents/execution_sandbox_agent/node.py
agents/execution_sandbox_agent/tools/execution.py
agents/execution_sandbox_agent/tools/mlir_embed.py
agents/execution_sandbox_agent/tools/benchmark.py
agents/execution_sandbox_agent/tools/mlir_utils.py
agents/judge_agent/node.py
agents/llm_reasoning_agent/node.py
agents/mutation_agent/
graph/pipeline.py
requirements.txt
docs/EXECUTION.md
docs/REPRODUCIBILITY.md
docs/INSTALL.md
docs/PIPELINE.md
runs/run_torch_L4_20260606_181557.txt
```

Old GitHub source was inspected directly and matches the key execution order described below. Historical repository contains a preserved run log, but that log is a **Torch Level 4 example, not the specific disputed Linalg pair**.

## 2. Goal and paper-level experiment scope

The accepted paper describes MLIRBench, a dataset and pipeline for compiler/hardware-aware semantic reasoning.

- 40 PyTorch seeds produced 1,768 execution-validated program pairs: **888 Linalg, 880 Torch**, across mutation levels L1–L5.
- Eight model reasoning evaluations produced **14,144 reasoning records**. A frontier model, Gemini 3.1 Pro, was added in the revision; its reported hard-label accuracy was 92.65% and aggregate score 74.18. Verify exact paper values against the final manuscript before using them in an AE report.
- Stages: Mutation Agent; MLIR validation; Execution Sandbox with IREE; input embedding/benchmarking; LLM Reasoning Agent; Judge; downstream tasks.
- Two downstream tasks were added in revision: benefit classification and semantic classification. Some experiments ran on NVIDIA A100 GPUs, so full downstream reproduction may not be possible on a local CPU or standard AEC machine. Provide a clearly documented lightweight path and distinguish it from full expensive reproduction.
- External LLM providers include OpenAI, Anthropic, Google, OpenRouter. Keys/costs need special AEC documentation; do not require keys for core dataset inspection or IREE verification.
- CPU execution: IREE LLVM CPU generic, vectorization/interleaving/unrolling disabled as listed below; deterministic seed 42 in historical runtime input generator.

Prior user goals: **one self-contained Docker container**, no Compose, no remote Torch-MLIR API service or GCP dependency. Current Mac is Apple Silicon, Docker tests run Linux AMD64 under emulation. The AE submission needs meaningful commands, clear README/docs, reproducibility reports, version/provenance metadata, and tractable tests.

## 3. NEW repository: dataset and Docker state as last observed

**Master dataset:**

```text
data/processed/MLIRBench/master/master.jsonl
```

This has exactly 1,768 lines/records. Important fields include:

```text
program_id
dialect
level
input_mlir
mutated_mlir
judge__outputs_diverge
pairs__status
judge__embedded_inputs_json
judge__semantic_explanation
```

There are additional benchmarking, status, reasoning, and evaluation fields. **Crucial:** `judge__outputs_diverge` is stored as a STRING, one of `'True'`, `'False'`, `'N/A'`, not a Python bool. Never use `is True`/`is False`; never rely on Python truthiness of `'False'` (truthy). Normalize explicitly:

```python
LABEL_MAP = {"True": "DIVERGE", "False": "EQUIV", "N/A": "ERROR_OR_UNAVAILABLE"}
```

Label counts:

| Stored label | Linalg | Torch | Total |
|---|---:|---:|---:|
| `'False'` (EQUIV) | 541 | 611 | 1,152 |
| `'True'` (DIVERGE) | 166 | 109 | 275 |
| `'N/A'` | 181 | 160 | 341 |
| **Total** | **888** | **880** | **1,768** |

Earlier breakdown suggested most N/A relate to expected invalid mutations and three compile errors, but verify actual status fields before deriving formal categories. Do not equate every N/A with a specific ERROR type without evidence.

Data subtrees under new repo include `data/raw/MLIRBench/...` and `data/processed/MLIRBench/...`, with categories such as `core`, `master`, `performance`, `mutation_provenance`, `reasoning`, `errors`, `evaluation`, and `model_arena`. Inspect actual files before depending on names. Raw original data must remain immutable.

**Docker image:** `mlirbench:dev`, running `--platform linux/amd64` on Apple Silicon host. Dockerfile based on `python:3.11-slim` with AMD64 platform; `mlir-19-tools` installed and Python packages pinned.

Verified inside image:

```text
Python:         3.11.18
numpy:          2.4.6
iree-compiler:  20241104.1068
iree-runtime:   20241104.1068
torch-mlir:     20261101
```

The **historical old repo requirements.txt pins NumPy 2.4.6 and both IREE packages 20241104.1068**, so these are already matched in Docker. Historic `docs/REPRODUCIBILITY.md` describes **Ubuntu 22.04, Python 3.10** for a *remote Torch-MLIR validation VM*, not necessarily the original local runtime. Historical Torch-MLIR exact version not established; `20261101` is the new Docker pin, not evidence for the dataset-creation build. Reinstalling IREE is not a likely fix for observed mismatches.

Compiler extra arguments (`_compile_extra_args('llvm-cpu')`):

```text
--iree-llvmcpu-target-cpu=generic
--iree-llvmcpu-loop-vectorization=false
--iree-llvmcpu-slp-vectorization=false
--iree-llvmcpu-loop-interleaving=false
--iree-llvmcpu-loop-unrolling=false
```

The `.dockerignore` reportedly excluded `dataset/raw` and `dataset/processed` but actual folders are `data/raw`/`data/processed`; build context had been ~1.87 GB. Fix carefully to exclude irrelevant materials without excluding required packaged dataset. Check `.dockerignore`, `COPY` and image contents. Need to rebuild after host source changes; smoke tests used read-only bind mounts, so `mlirbench:dev` may still have older files.

## 4. Historical pipeline behavior discovered

In old repo `agents/execution_sandbox_agent/node.py`, both in the June 6 commit and later:

```python
result_a = compile_and_run_mlir(program_a, dialect=dialect)
result_b = compile_and_run_mlir(program_b, dialect=dialect)
# benchmark on compiled modules
self_containedA, embed_inputs_programA = embed_inputs(program_a)
self_containedB, embed_inputs_programB = embed_inputs(program_b)
# THEN compare_outputs(output_a, output_b)
```

`compile_and_run_mlir` uses `run_compiled`, which generates inputs from the `@main` argument signature using `np.random.default_rng(42)` unless explicit `input_values` are supplied. If argument signatures/order match, each side gets equal random values despite separate generation. The later `embed_inputs(program_a)` / `embed_inputs(program_b)` generates *different input tensors* from each MLIR's constants and is used for metadata / LLM prompt, **not directly as the runtime inputs for historical ground-truth comparison in this source**. This is a crucial distinction. Do not claim the original agent directly compared independently embedded programs without discovering another actual generation pathway.

`embed_inputs` in `tools/mlir_embed.py` independently derives values from constants in each source; the original and mutated versions may contain different constant order or sets even if runtime operations are equivalent. It returns a transformed MLIR program and a dict of input values. `node.py` historically did not execute those returned self-contained programs in the code inspected.

`compare_outputs()` uses `np.array_equal` for numeric arrays and Python equality elsewhere. It describes exact matching, but NumPy value equality is not bitwise equality in all edge cases (e.g. +0.0 versus -0.0). Investigate NaN, infinities and bitwise claims rather than changing semantics implicitly.

Original report `Final Project Report.pdf` (69 pages, available in earlier ChatGPT session, may not be on local disk) says the original experimental Linalg execution strategy embedded deterministic constants into the program, whereas Torch used external runtime inputs due to a prior IREE canonicalizer issue. This is at least superficially inconsistent with the checked-in historical `node.py` execution order. Possible separate offline dataset-generation scripts or earlier workflow are still unlocated. **Resolve from Supabase raw execution payloads and archived scripts, not from assumptions.** The report does not pin exact original LLVM/MLIR/IREE/Torch-MLIR versions.

Historical preserved GitHub log `runs/run_torch_L4_20260606_181557.txt` shows `execution_results.programA_output` and `programB_output` identical and `outputs_diverge: false`, including value `2.2557523250579834`, and separate `embed_inputs_program_a`/`b` fields. The run is Torch L4 and does **not** contain `seed1_linalg_prog7_L1_R2`. Missing `langchain_google_genai` caused its semantic explanation to be unavailable. It confirms existence of full nested original outputs in historical payloads, which the flattened current master may lack.

## 5. Code changes already applied to NEW repository (very important)

Initially, new `agents/execution_sandbox_agent/tools/execution.py` had functions `run_compiled` and `compile_and_run_mlir` that generated seed-42 inputs. They were patched to support an optional `input_values: dict | None = None`, checking names, shapes and dtypes and passing explicit runtime values when supplied. Also fixed `@main` regex for `func.func @main(...)`. This modified host file was mounted into Docker during tests because image was not rebuilt.

A later **shared-input patch was successfully applied** locally. It created:

```text
agents/execution_sandbox_agent/tools/pair_ground_truth.py
```

and changed:

```text
agents/execution_sandbox_agent/node.py
```

The patch introduced `execute_pair_shared_inputs(program_a, program_b, dialect, seed=42)` and rewired the execution node to use it. It:

1. Parses argument specs and SSA names from both `@main` signatures.
2. Raises `ValueError` when ordered specs differ.
3. Calls `embed_inputs(program_a)` **once**, using generated input values from original program as the pair's shared inputs; falls back to seed-42 random values for missing inputs.
4. Maps the same arrays by positional argument to both programs even if argument SSA names differ.
5. Calls `compile_and_run_mlir(..., input_values=...)` for A and B.
6. Records input provenance, a SHA-256 fingerprint of serialized values, and matching per-program inputs.
7. Replaces independently generated embedded-input metadata forwarded to LLM with those shared values.

The patch printed successful syntax checks:

```text
PASS: Shared-input execution module created.
PASS: Execution agent now uses the shared-input procedure.
PASS: Downstream input metadata uses those same inputs.
PASS: Input provenance is recorded.
PASS: Both modified Python files compile.
Original dataset has not been modified.
```

The helper was then tested inside Docker by mounting host `execution.py` and `pair_ground_truth.py`, and two examples compiled/executed with identical shared arrays:

```text
seed1_linalg_prog7_L1_R2  published DIVERGE, observed EQUIV
  input source: original_program_embedded_inputs
  identical inputs: True
  A/B compilation: success
  A/B execution: success

torch_seed4_prog40_T1_R1 published DIVERGE, observed EQUIV
  input source: original_program_embedded_inputs
  identical inputs: True
  A/B compilation: success
  A/B execution: success
```

**Do not consider this final or comprehensive.** Important design flaws to evaluate in the patch:

- It uses original-derived embedded input values for **Torch**, despite report saying Torch was originally passed external runtime values. These generated values may violate operation-specific preconditions.
- Its fallback generic integer generation includes invalid indices for ops such as `torch.aten.gather`.
- It uses a **single** pair-level input sample to decide EQUIV/DIVERGE, which is insufficient for semantic-equivalence claims.
- For mutations that change `@main` argument signatures, it raises `ValueError`, currently causing an unhelpful execution failure. Need explicit policy for incompatible signatures or a valid input adapter.
- It builds metadata with argument names rather than a canonical position-only schema; ensure consistent mapping for renamed SSA arguments.
- It may invoke `embed_inputs` on `@main` with unusual signatures and fails/falls back unpredictably; check shapes/dtypes and support scalars.
- It creates a hash only of JSON-serialized values; consider storing dtype/shape + seed + policy version + platform/toolchain with the fingerprint.
- Host `node.py` has **not been exercised end-to-end** with the patched code inside rebuilt Docker. Only helper smoke tests and host syntax compilation passed.
- The old `_generate_deterministic_inputs` generates random booleans, normal floats, integers `[-100, 100)`; the correctness domain of each operation must be respected.
- Current `compare_outputs` should be audited for float/NaN/signed zero exactness and compared against paper claims; don't silently change comparison criterion.

Review, harden or replace this helper based on evidence. Keep changes source-controlled and add tests.

## 6. Test history and specific results

### Initial sample

A script `scripts/validate_dataset_sample.py` selected 20 deterministic records (both dialects, EQUIV/DIVERGE L1–L4, ERROR L5). Earlier run reported **16 PASS / 4 FAIL**, with four published DIVERGE cases observed EQUIV using shared seed-42 values:

```text
seed1_linalg_prog7_L1_R2
seed1_linalg_prog10_L3_R4
seed1_linalg_prog10_L4_R4
torch_seed4_prog40_T1_R1
```

Six hand-picked EQUIV/DIVERGE/ERROR examples also had passed earlier. Verify the validator script, its actual execution paths, and mount requirements. Do not treat passing tests as proof of mathematical equivalence.

### Disputed first Linalg pair and input embedding investigation

`seed1_linalg_prog7_L1_R2` has identical original/mutated signature `tensor<3x4xf32>`. With runtime seed-42, both output:

```text
[False, 2.2557523250579834, [False, False, False], [False, False, False, False], 1.0, 0.0833333358168602]
```

With **stored original embedded inputs** supplied identically to A and B, both output:

```text
[False, 799999.875, [False, False, False], [False, False, False, False], 1.0, 0.0833333358168602]
```

Stored original input and original-generated embedded input matched:

```json
{"%arg0": [[0.0, 11.0, 11.9, 12.0], [12.1, 13.0, -1.0, -0.1], [0.1, 1.0, 799999.0, 799999.9]]}
```

Mutated `embed_inputs(mutated_mlir)` independently produced:

```json
{"%arg0": [[0.0, -1.0, -0.1, 0.1], [1.0, 799999.0, 799999.9, 800000.0], [800000.1, 800001.0, 11.0, 11.9]]}
```

Executing **independently embedded** versions generated:

```text
original: max value 799999.875
mutated:  max value 800001.0
```

Exactly matches the published semantic explanation, which erroneously/at least misleadingly attributes the different max output to operation reorder/non-associative floating-point max. The actual input values differ! This is strong causal evidence for a **potential explanation/ground-truth provenance defect**, but historical checked-in execution node does not run embedded programs for ground truth. Do not claim confirmed root cause for original 1,768 dataset until original raw execution payloads are inspected.

An audit of `embed_inputs(original)` versus `embed_inputs(mutated)` across the dataset gave:

```text
Linalg: 704 same, 184 different out of 888 (20.72%).
  published DIVERGE:  96 same, 70 different
  published EQUIV:   434 same, 107 different
  published N/A:     174 same, 7 different
Torch: 728 same, 152 different out of 880 (17.27%).
  published DIVERGE:  85 same, 24 different
  published EQUIV:   483 same, 128 different
  published N/A:     160 same, 0 different
Total: 336/1768 have generated-input differences.
94/275 published DIVERGE pairs have generated-input differences.
```

This is a *risk indicator*, **not** proof that all 94 labels are wrong. For Torch, those independently generated values may not have been used during historical execution at all.

### Corrected 12-record sample audit, with shared inputs derived from original `embed_inputs`

The first audit accidentally selected only four records because labels are strings. After correcting selection to `'True'`/`'False'`, all 12 were tested:

| ID | Published | Observed with shared original-derived embedded inputs | Result |
|---|---|---|---|
| `seed1_linalg_prog7_L1_R2` | DIVERGE | EQUIV | Mismatch |
| `seed1_linalg_prog10_L3_R4` | DIVERGE | DIVERGE | Match |
| `seed1_linalg_prog10_L4_R4` | DIVERGE | EQUIV | Mismatch |
| `torch_seed4_prog40_T1_R1` | DIVERGE | EQUIV | Mismatch |
| `seed2_linalg_prog16_L5_R2` | EQUIV | EQUIV | Match |
| `seed1_linalg_prog1_L1_R5` | EQUIV | EQUIV | Match |
| `seed2_linalg_prog16_L4_R4` | DIVERGE | EQUIV | Mismatch |
| `seed2_linalg_prog13_L4_R4` | DIVERGE | EQUIV | Mismatch |
| `torch_seed1_prog5_T1_R5` | EQUIV | EQUIV | Match |
| `torch_seed3_prog21_T4_R1` | EQUIV | EQUIV | Match |
| `torch_seed1_prog4_T2_R3` | DIVERGE | EQUIV | Mismatch |
| `torch_seed2_prog16_T2_R5` | DIVERGE | cannot compare signatures | Exception |

**Summary: 5 MATCH, 6 MISMATCH, 1 input-specification exception.**

This is a deliberately targeted, nonrandom representative diagnostic including four known failures. It is **not** an estimate of overall dataset error rate.

### Five-seed replay of six mismatch cases with generic same-seed runtime inputs

Each original and mutated MLIR compiled once; both received same deterministic seed per replay; seeds `[0, 1, 42, 123, 2026]`. Results:

```text
seed1_linalg_prog7_L1_R2    EQUIV for all 5
seed1_linalg_prog10_L4_R4   EQUIV for all 5
seed2_linalg_prog16_L4_R4   EQUIV for all 5
seed2_linalg_prog13_L4_R4   EQUIV for all 5
torch_seed1_prog4_T2_R3     DIVERGE for all 5
torch_seed4_prog40_T1_R1    EQUIV for all 5
```

Therefore, one Torch case is genuinely *witnessed* as divergent under those shared random inputs despite EQUIV under original-derived embedded inputs. Five others remain **not reproduced** with those five random seeds, not proven equivalent.

### Important read of unified diffs for five unreproduced cases

1. `seed1_linalg_prog7_L1_R2`: moves independent operations, constants and `tensor.empty` statements. The recorded explanation claims max-reduction nonassociativity, but runtime output difference observed only under *different* independently embedded input arrays. Investigate classification/explanation.
2. `seed1_linalg_prog10_L4_R4`: extracts operations into helper functions such as `@compute_bounds`, `@compute_reductions`, `@compute_min`. Recorded explanation claims `%arg0` runtime input tensor constants changed, but `%arg0` is a function argument; changes in independently generated input values are not an MLIR semantic change.
3. `seed2_linalg_prog16_L4_R4`: extracts `@compute_normalized`, `@compute_statistics`, `@compute_entropy`. Recorded explanation again attributes difference partly to changed input data, despite `%arg0` being runtime argument. Check for meaningful operation differences, e.g. call order or actual changed constants, rather than accepting the explanation.
4. `seed2_linalg_prog13_L4_R4`: extracts `@split_geometric`, `@split_validation` and changes calls, but explanation claims runtime tensor values changed. Initial diff shows refactoring, not a literal dense `%arg0` constant in `@main`.
5. `torch_seed4_prog40_T1_R1`: primarily SSA renaming, including renaming the two main function arguments. SSA-only renaming should not change semantics. Original program contains `torch.aten.gather` from shape `[6,3]`, with index tensor of shape `[6]`; indices for gather along dimension 1 must be within `0..2`. Historic generic generator for `si64` uses random integers `[-100, 100)`, which may be out of bounds. Previous outputs contained very large / nonsensical values. The recorded explanation attributed small differences to nondeterministic floats despite otherwise equivalent SSA refactor. This is suspicious; must run with **valid common gather indices** before making conclusions.

**Most recent pending test** from the previous thread was a targeted Torch run for `torch_seed4_prog40_T1_R1` with valid shared `indices=[0,1,2,0,1,2]` and random float `[6,3]` for seeds 42 and 123. This test has **not been reported as executed**. Do it next, then inspect outputs with care. The patched `compile_and_run_mlir(... input_values=...)` is required.

## 7. Root-cause hypotheses and what is/is not established

**Established:**

- Matching historical NumPy/IREE package pins did not eliminate label mismatches.
- The new environment can compile and run these examples; four initial DIVERGE labels observed EQUIV with seed-42 shared inputs.
- Different independently embedded inputs can reproduce the exact difference claimed in one Linalg semantic explanation.
- The historical checked-in execution agent executes seed-42 `compile_and_run_mlir` **before** calling `embed_inputs` for metadata, so the old code alone does not prove independent embedding caused published labels.
- The dataset's `'True'/'False'/'N/A'` fields are strings, creating an easy audit bug.
- Valid runtime input domains matter for `torch.aten.gather` and possibly division/log/sqrt/etc.; generic random inputs can be invalid.
- Some labels are input-dependent; one Torch DIVERGE pair became DIVERGE for all random seeds while EQUIV on original-derived embedded values.
- The historical explanations for several Linalg cases treat varying `%arg0` runtime inputs as if they were changed MLIR constants.

**Not established:**

- That every published DIVERGE mismatch is a mislabeled sample.
- That the old GitHub execution path exactly matches the original offline full-dataset creation path.
- Exact Torch-MLIR version/compiler commit for the published experiments.
- Validity, determinism, or correctness of all original runtime outputs.
- That a single code edit will retroactively make all 1,768 original ground-truth labels reproduce.
- That finite equal-output tests prove semantic equivalence.

**Likely high-value next evidence:** original raw Supabase JSON export with nested `Agents.Execution_Agent.execution_results.programA_output` and `programB_output` and independently embedded inputs for each pair; any archived offline generation scripts or logs from `Active/MLIRBench/Experiments`, `Results`, `Code`, or `Data/Raw`.

## 8. Next tasks in priority order

### P0. Safety and ground-truth provenance

1. Inspect current `git status`, `git diff`, untracked files, Dockerfile, README and new helper; avoid losing existing code. Use branch/commit discipline.
2. Inspect Active `Data/Supabase/supabase_export/json/mlirbench_master.json` and full nested records. Find disputed IDs by `program_id`, compare original raw output A/B, input metadata, labels, status, compiler/version metadata and timing. Determine if original labels were actually computed from different embedded values or from shared runtime inputs.
3. Inspect historical alternative offline dataset scripts/logs, and paper/revision methodology in `Active/MLIRBench/Papers`. Reconcile code versus report without inventing facts.
4. Complete the **pending Torch valid-index replay** for `torch_seed4_prog40_T1_R1`. Save exact input tensor, dtype, shape, seed, compiler flags, A/B outputs and result. Compare with recorded log if available.
5. Characterize the altered `@main` signature in `torch_seed2_prog16_T2_R5`, and define correct behavior for signature-incompatible mutations.

### P1. Fix input-generation and replay contract

Implement a documented **canonical pair-level execution/replay API**, with tests:

- Determine/validate signature compatibility and input argument mapping. Either safely adapt supported signature differences or report `INCOMPARABLE_INPUT_SIGNATURES` distinct from semantic DIVERGE.
- Derive shared input values **once per pair per test**, not separately; metadata must contain the **actual values run**, types, shapes, seed, domain policy, checksum, code/environment/compiler versions and output fingerprints.
- Support replay of **recorded original input values** when available, independent of generated test inputs.
- Support deterministic suites with multiple seeds and *operation-aware valid input domains* (gather/scatter indices, division/log domains, shape constraints, data-dependent indexing, etc.). Avoid arbitrary invalid tensors. Do not assume all operations can be handled generically; report unsupported preconditions explicitly.
- Separate three questions: (a) published historical record replay, (b) newly defined canonical shared-input validation, (c) metamorphic/semantic analysis. Do not claim they are equivalent.
- `DIVERGE` is justified if at least one **valid shared input** gives reliably different outputs; no witnessed difference should be reported as `NOT_REPRODUCED`/`UNVERIFIED` rather than automatically `EQUIV`.
- Distinguish compile error, runtime error, timeout, invalid input, nondeterminism, and incompatible signature. Inspect exact float semantics and paper's bit-exact claim.
- Benchmark only when appropriate, with defined warmup/repetition counts; avoid burning 10 s warmup × 1,768 on a quick validator.
- Produce machine-readable JSONL and aggregate CSV/Markdown with per-ID observed verdict and mismatch category; published records remain immutable.
- Add unit/integration tests covering SSA renames, identical programs, genuine changed ops, changed signature, valid/invalid gather indices, mismatch type, old string labels, stable deterministic replay, and nonfinite floats.

### P2. Scale checks without falsely changing results

- Re-run targeted smoke, then balanced stratified sample across dialects, levels L1–L5, labels EQUIV/DIVERGE/ERROR, plus known failures. Specify selection criteria and sample IDs.
- Cache compilations when possible and keep execution resource limits. Log total successes/failures. Run full dataset only after tests succeed and costs/time are understood.
- If historical labels cannot be reproduced, publish a transparent discrepancies report, and discuss whether a separately versioned corrected dataset requires updating derived judge metrics. Do not overwrite published `master.jsonl` or change paper claims by default.

### P3. Full AE packaging

- Ensure **all** relevant `agents`, `graph`, `services`, `scripts`, dependencies and data are tracked in Git. `node.py` was untracked; audit all files.
- Review `Dockerfile`, `.dockerignore`, mounts and `COPY` paths, build AMD64 image; verify functionality without manually mounting unpublished local changes.
- Provide a one-container `docker build` and CLI `docker run` path with `--help`, `check`, `dataset-info`, sample verification, and a minimal single-pair replay. Possibly `python run.py check` if appropriate to architecture.
- Keep Torch-MLIR verification local in image; do not require original remote API to run core tests. Document any historical validation differences.
- Make LLM/Judge API steps optional; no automatic external calls or credentials in repository; provide recorded predictions and deterministic offline scoring if possible.
- Document A100 downstream tasks separately: datasets, code, checkpoints, hardware requirements, estimated cost and the feasible limited demonstration. Verify correct paper reported F1s and settings.
- Provide proper `README.md`, `ARTIFACT.md` or `REPRODUCIBILITY.md`, quick start, output examples, version table, known limitations, data provenance and expected results. Produce AEC tester-friendly commands and status report.
- Rebuild and test image from a clean Git checkout; compare actual outputs to documentation.

## 9. Useful investigation entry points

**New repo code paths to inspect first:**

```text
Dockerfile
.dockerignore
requirements.txt
run.py
scripts/validate_dataset_sample.py
agents/execution_sandbox_agent/node.py
agents/execution_sandbox_agent/tools/execution.py
agents/execution_sandbox_agent/tools/mlir_embed.py
agents/execution_sandbox_agent/tools/pair_ground_truth.py
agents/execution_sandbox_agent/tools/benchmark.py
data/processed/MLIRBench/master/master.jsonl
```

**Original archive files to locate/read:**

```text
Active/MLIRBench/Code/MLIRBench/agents/execution_sandbox_agent/node.py
Active/MLIRBench/Code/MLIRBench/agents/execution_sandbox_agent/tools/execution.py
Active/MLIRBench/Code/MLIRBench/agents/execution_sandbox_agent/tools/mlir_embed.py
Active/MLIRBench/Code/MLIRBench/requirements.txt
Active/MLIRBench/Data/Supabase/supabase_export/json/mlirbench_master.json
Active/MLIRBench/Experiments/
Active/MLIRBench/Results/
Active/MLIRBench/Papers/
Active/MLIRBench/Downstream Tasks/
```

**Prior docker test invocation pattern:**

```bash
docker run --rm -i --platform linux/amd64 \
  -v "$PWD/data/processed/MLIRBench/master:/dataset:ro" \
  -v "$PWD/agents/execution_sandbox_agent/tools/execution.py:/app/agents/execution_sandbox_agent/tools/execution.py:ro" \
  -v "$PWD/agents/execution_sandbox_agent/tools/pair_ground_truth.py:/app/agents/execution_sandbox_agent/tools/pair_ground_truth.py:ro" \
  --entrypoint python mlirbench:dev - <<'PY'
# Python test code here
PY
```

These mounts were needed only because Docker image had not been rebuilt. The final artifact should not depend on them.

## 10. What you should do immediately, as Codex

**Do actual local engineering, not more speculative discussions.**

1. Confirm new repository path and Git state, inventory code/tests, check whether the original archive and Supabase export paths exist. Give a concise audit of what you found.
2. Inspect original raw records and execution output provenance for the known mismatches. Determine what evidence is available for historical label formation. Explicitly label inference versus fact.
3. Inspect and test the already applied `pair_ground_truth.py` patch. Complete valid-gather-index targeted replay and investigate mismatched input signature. Fix defects with tests, not just local scripts.
4. Propose and implement a clean, reproducible **historical replay + canonical validation** split without overwriting data. Preserve the currently working image until a new image passes tests.
5. Then progressively prepare Docker packaging, git tracking and AEC documentation; report exact files changed, tests executed, outcomes, remaining mismatches, and user decisions needed.

**Success criterion:** A reviewer can build one image, inspect 1,768 published records, replay a documented deterministic subset correctly under a clearly stated input policy, and see a transparent report explaining any historical label discrepancies. No fabricated equivalences, no hidden label edits, no dependence on cloud validation, and no missing untracked code.

