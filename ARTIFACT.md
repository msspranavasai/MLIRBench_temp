# Artifact execution contract

This artifact is under validation. It preserves published results and adds an independently versioned audit path. It does not claim that all historical labels have been reproduced or that the complete paper experiments are currently packaged.

## Build and offline checks

Use the commands in README.md. The image targets Linux AMD64; Apple Silicon runs it under emulation. Python 3.11.18, NumPy 2.4.6, IREE compiler/runtime 20241104.1068, and Torch-MLIR 20261101 are pinned. The latter is an artifact dependency, not an established historical compiler version. Debian packages are obtained from the Trixie package repositories; the existing Python base image is pinned by digest; a fully frozen apt snapshot remains to be established.

The Dockerfile explicitly copies source, tests, documentation, a deterministic compressed bundle of processed JSON/JSONL data, and historical execution evidence. The build verifies compressed and uncompressed hashes before unpacking the bundle. Working raw/processed copies remain untouched; the bundle provides a practical Git distribution for files exceeding GitHub’s normal file-size limit. Raw data, duplicate processed CSVs, local reports, virtual environments, Git metadata, and credentials are excluded. All processed JSONL views remain available, including recorded reasoning and evaluation data. The published master SHA-256 is `84718efccae10e5bbd66fd7c4455ca9f22a8957c968151465b830855335a0541`.

## Canonical validation

```sh
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD/results/aec:/results" mlirbench:aec \
  validate --sample --output /results/sample.jsonl
```

Selection is stable and lexicographic within every available dialect/level/label stratum, augmented with eight named diagnostics. The JSONL records all IDs. `--all` explicitly requests the complete published dataset; run a sample first. `--seeds 42 123` selects a smaller suite. `--timeout 30` bounds each runtime child process; compilation has a separate 120-second bound. No benchmarks run during verification.

`valid-domain-v2` generates positive floating-point values in [0.25, 2), small positive integers, random booleans, and bounded gather indices. Constraints are intersected across both programs. A restricted NumPy Torch preflight checks actual intermediate domains, gather rank/shape/bounds, and division/log/sqrt inputs. It never supplies the expected program outputs. Linalg currently supports static tensor operations with constant nonzero division denominators; helper functions, dynamic/scalar indexing, log/sqrt/power and integer-domain operations require further analysis and are reported as unsupported. This is a limited domain checker, not an arbitrary-MLIR verifier or solver. It deliberately rejects programs beyond its support. Additional operation support needs tests before inclusion.

Each program is compiled once per pair. Each input is mapped by argument position (SSA names may differ), then executed at least twice per program in independent child processes. Reports contain exact input bytes, dtype, shape, seed, policy, source/code/package/platform metadata, compiler flags, compiled-module fingerprints, typed output bytes and fingerprints. Dtype changes are incomparable; no implicit f32/f64 adapter is used.

`typed-exact-value-v1` requires equal output structure, shapes, dtypes and numerical values without tolerance. Positive/negative zero compare equal. Byte fingerprints separately detect repeat instability, including zero sign changes. Nonfinite outputs receive `NONFINITE_OUTPUT`, never DIVERGE. The historical `compare_outputs` function is retained unchanged: its `numpy.array_equal` semantics are value equality, not strict bit identity. Historical NaN handling can depend on whether outputs are homogeneous numeric arrays or heterogeneous Python containers.

Verdicts distinguish `DIVERGE`, `NO_DIFFERENCE_OBSERVED`, incompatible/unsupported signatures, invalid/unsupported inputs, compilation/runtime errors and timeouts, nondeterminism and nonfinite outputs. A deterministic valid differing trial is a witnessed counterexample under this policy. Equal finite trials do not establish equivalence. A witnessed counterexample is retained even if another trial is unsupported; all trial results remain visible.

The CLI writes JSONL plus CSV and Markdown summaries. Existing paths are rejected. A published DIVERGE with all equal trials is `NOT_REPRODUCED`. Published EQUIV receives `NO_COUNTEREXAMPLE_OBSERVED` or `COUNTEREXAMPLE_FOUND`, never a new equivalence proof. N/A is kept separate; only explicitly expected-invalid mutations with original compilation success and mutation compilation error receive `EXPECTED_MUTATION_COMPILE_ERROR`. CLI exit zero means the audit completed, not that all labels matched. Inspect verdicts and summaries.

## Recorded inputs and historical evidence

`data/replay/historical_execution.jsonl` projects 1,768 uniquely matched records from the original 1,804-row `execution_agent_unified_rows.csv`. It includes original output strings, input metadata, source filenames, runtime settings, and CSV/row/source-program SHA-256 provenance. Published datasets are unchanged. Regeneration from the read-only original CSV is possible with `scripts/audit_execution_provenance.py`; it refuses overwrite.

A typed positional input list saved in a report's `trials[n].inputs` can be written to JSON and replayed:

```sh
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD/results/aec:/results" mlirbench:aec \
  validate --id torch_seed4_prog40_T1_R1 --inputs /results/inputs.json \
  --output /results/replay.jsonl
```

Dtype, shape and values must match the signature. Replay applies the same domain checks. `--stored-metadata` instead tests the historical original embedded-input metadata as common inputs. These values are not asserted to be the actual historical runtime inputs. It may report invalid inputs, including negative gather indices.

The separate historical hypothesis command deliberately reconstructs independent Linalg inputs and compares both results to the original CSV:

```sh
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD/results/aec:/results" mlirbench:aec \
  historical-candidate --id seed1_linalg_prog7_L1_R2 \
  --output /results/historical.json
```

Its result is historical output reproduction, not a semantic verdict. Unsafe historical Torch inputs are not executed. The archived generation scripts referenced by the CSV have not been located; exact historical toolchain provenance remains incomplete.

## Model evaluation, performance, and downstream work

The default image never requests model credentials. The interactive historical pipeline is opt-in via `pipeline`; new execution results attach a canonical suite verdict and never assign EQUIV from finite testing. Semantic explanation calls require an explicit payload option. Benchmarks are skipped in this path because the historical type-only benchmark inputs differ from validation inputs. This behavior is intentionally distinct from the preserved historical data and scoring.

The processed reasoning, evaluation and model-arena views support offline inspection. `score-recorded` deterministically recomputes label accuracy against the immutable master and summarizes stored judge scores with both available-row and full-dataset denominators. It checks duplicate IDs and exposes any disagreement between recorded and published ground truth. It does not regenerate subjective judge scores. New provider calls are optional and potentially paid; no API keys are packaged.

The evaluation view contains 15,907 rows: eight models with 1,768 rows each and Gemini with 1,763. Five Gemini model failures have no evaluation rows. Gemini hard-label accuracy over all 1,768 pairs is 92.6471%; its mean recorded judge score over 1,763 available rows is 74.1806. Counting five missing scores as zero gives 73.9708 over all pairs. The historical merge script explicitly identifies these five as MODEL_FAILURE. These denominators explain the stored numbers; no paper values or source records are changed.

The old reference project contains `Downstream Tasks/Downstream-Task-2.ipynb`; the Experiments and Results directories contain no source files found by this audit. Full downstream training, benefit classification code, checkpoints, split provenance, GPU dependencies and paper-table reproduction remain to be located and packaged. GPU/A100 experiments are not reproduced by these CPU validation commands. No downstream F1 or accuracy claims have been modified.

Before AE submission: extend domain coverage, resolve historical discrepancies, validate on native AMD64, freeze remaining dependency provenance, make a clean version-controlled build, add verified scoring/downstream commands, and have a fresh evaluator follow the instructions.
