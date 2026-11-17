# Reviewer instructions

## Default: no paid model calls

Prerequisites: Docker with Linux containers and Python 3 (standard library only for host-side build-input checks) and disk space for the compiler image, published data and generated stage artifacts. Windows users can use WSL2/Bash. The launcher pulls one versioned multi-platform image containing the compiler/runtime and original optional agent dependencies. Its native platform is selected automatically. Image acquisition needs network access; the archived experiment container runs with networking disabled. A private package requires registry authentication until the author makes it public. An explicit source build is available with AE_IMAGE_SOURCE=build.

```bash
./ae.sh quickstart
```

This verifies frozen input hashes, runs unit/integration tests, reconstructs the archived mutation/execution/reasoning/judge stages, recalculates objective correctness and score totals, computes paper tables, replays downstream saved-prediction metrics, and runs two real CPU execution demonstrations. The default does not call any model API. This archived-response route is the planned AE reviewer path: no API account, credential, credit purchase or remote API service is required. Saved model responses and judge observations are declared inputs; reviewers rerun the transformations, scoring/aggregation and CPU validation themselves. Optional live-agent source is included for additional experiments and is not a required paid reproduction step. Follow the printed START_HERE.md; its pipeline/REPORT.md identifies each stage and its data source.

Generated artifacts are written to ../MLIRBench-AE-runs, outside the code repository. Set AE_OUTPUT_ROOT to change that host folder. /results in commands maps to that folder. Every run requires a new output directory; prior results are never overwritten. Successful prerequisites are checkpointed, and completed fresh cases are saved individually. Use the explicit continuation command below to recover a stopped fresh stage without repeating its committed cases. No console-log files are created.

## Reproduce one stage at a time

```bash
./ae.sh pipeline --stage mutation --output /results/01-mutation
./ae.sh pipeline --stage execution --input /results/01-mutation --output /results/02-execution
./ae.sh pipeline --stage reasoning --input /results/02-execution --output /results/03-reasoning
./ae.sh pipeline --stage judge --input /results/03-reasoning --output /results/04-judge
./ae.sh pipeline --stage metrics --input /results/04-judge --output /results/05-metrics
```

Replay is the default. A requested stage loads or replays its prerequisites and writes a self-contained stage checkpoint. Use --id PROGRAM_ID and --model MODEL_KEY to select a subset. Subset results are clearly marked and are not presented as complete paper tables.

Mutation replay restores the published pairs and mutation metadata. Execution replay restores archived observations, rather than claiming a new compiler run. Reasoning replay restores the actual saved model answers. Judge replay retains saved subjective result/explanation ratings and recalculates correctness and weighting. Missing observations are disclosed. Missing judge observations are recomputed from the selected reasoning records without a matching judge record, including metrics resumes and ID/model subsets. Stored counts from older runs are not trusted. An explicitly unscored judge record is retained as an observation, rather than counted as missing. Raw null predictions retain their original values alongside the historical failure-to-ERROR normalization. Recomputed and archived total scores remain side by side.

The total score is 100/3 × (label correctness + result_accuracy/10 + explanation_rating/4). The archived scoring fields can have finite precision; the new totals preserve their raw arithmetic and disclose differences. The existing author-defined Table 4 agreement policy remains at 0.15 points on a 0–100 scale, requiring all 45 cells and unchanged model/level orderings. Other checks retain their criteria. This is not an AE badge decision or an execution-output tolerance.

Source files, record contents, program pairs and parent stages have SHA-256 identities. Resuming verifies the saved artifacts and their dependency links. New programs or responses cannot quietly reuse old downstream scores. Reconstructed archive lineage is not proof that the unavailable original runtime arguments have been recovered.

## Actual execution without API charges

```bash
./ae.sh pipeline --stage execution --mode fresh --id seed1_linalg_prog1_L1_R1
./ae.sh pipeline --stage execution --mode fresh --id torch_seed1_prog10_T1_R1
```

The execution agent compiles and executes both programs with positional shared inputs, operation-aware domain checks, five deterministic seeds and two fresh-process repetitions per program/input. Exact typed outputs are compared. EQUIV means observed equality across this suite, not universal equivalence. Unsupported input domains, timeouts and nonfinite outputs remain unscored. Fresh execution is kept separate from the archived observations used for paper metrics.

## Optional live-agent code (outside the required reviewer route)

The optional live implementation is provided for users who independently choose and arrange provider access. The submission does not provide or promise API keys, credits or an API access server, and evaluators are not expected to supply them for the required route. Its runtime guards still require explicit --allow-paid-api, a valid configuration and the relevant credentials before any model call; no credentials are distributed with the code. The launcher enables container networking only for this explicit opt-in and forwards named provider environment variables or an explicitly supplied AE_ENV_FILE. Credentials are not copied into the image or saved in run artifacts.

```bash
# Optional additional experiment using independently arranged access; may incur API costs.
./ae.sh pipeline --mode fresh --id seed1_linalg_prog1_L1_R1 \
  --model gpt-4.1-mini --allow-paid-api

# Continue from a fresh execution checkpoint, using its actual inputs.
./ae.sh pipeline --stage reasoning --mode fresh --input /results/FRESH_EXECUTION_RUN \
  --model gpt-4.1-mini --allow-paid-api
```

The runner validates all requested live models, credentials and request budgets before starting any agent. See [fresh API configuration](docs/live-configuration.md) for exact provider IDs, stage-specific credentials, limits and the limits of local preflight. Each fresh run records the resolved configuration without credential values.

Fresh runs default to one pair and one reasoning model. --all explicitly permits a larger selected experiment. Mutation uses the original gpt-5-mini configuration (MLIRBENCH_MUTATION_MODEL overrides it); judging uses the original judge implementation/configuration (OPENAI_MODEL overrides it). Provider/model availability and a real paid request have not been validated in this no-cost preparation run. Failures remain failures; fresh scores are never filled with archived values. Invalid raw judge JSON, missing component fields, nonnumeric/nonfinite/out-of-range ratings and inconsistent component maxima are rejected before scoring. The raw invalid response and its hash are retained in the failed case observation. No fallback rating or clamped value is substituted. The judge receives the full recorded mutation rule and inspection metadata, or an explicitly identified packaged-catalog rule for an archived pair; unknown or inconsistent rules are refused. These corrections affect optional fresh judging only, not archived ratings or paper aggregation.

To use a new MLIR source rather than a published pair:

```bash
./ae.sh pipeline --mode fresh --source /absolute/path/program.mlir \
  --dialect linalg --level 1 --requirement L1_R1 \
  --model gpt-4.1-mini --allow-paid-api
```

The source is mounted read-only. Fresh mutations get content-derived identifiers. Fresh/subset metrics are reported separately from the full archived paper comparison. To proceed from fresh execution to reasoning and judging, rerun those stages explicitly; archived outputs are refused when their upstream stage has changed.

## Continue a stopped fresh run

New fresh runs save each completed case in `cases/` immediately, before starting the next case. Failure/interruption types and attempt history are recorded separately from successful stage records; no score is invented for a failed call. Stage artifacts and manifests are committed with atomic replacement and filesystem sync. Partial stages are not exposed as completed stage outputs to metric replay.

Use a new output directory and repeat the original stage, model, selection and request-limit settings:

```bash
# Example: the stopped command originally requested a multi-pair reasoning stage.
./ae.sh pipeline --stage reasoning --mode fresh \
  --input /results/STOPPED_REASONING_RUN --output /results/CONTINUED_REASONING_RUN \
  --model gpt-4.1-mini --all --allow-paid-api --resume-incomplete

# Example: continue an original full fresh pipeline (default stage: all).
./ae.sh pipeline --mode fresh \
  --input /results/STOPPED_FULL_RUN --output /results/CONTINUED_FULL_RUN \
  --model gpt-4.1-mini --allow-paid-api --resume-incomplete
```

Only continue a stopped run. Keep `--all` if the original selection contained multiple pairs. Omit `--id` to retain the saved selection, or repeat exactly the original IDs; changing the selection is refused in continuation mode. A custom source is restored from the saved case plan, so omit `--source`, `--dialect`, `--level` and `--requirement` when continuing it. Self-contained fresh checkpoints and custom-source continuations do not require published views at runtime. Reusing an existing image as described below also avoids rebuilding from unrelated host bundles.

`--resume-incomplete` verifies the plan, case records, parent identities, source fingerprint and resolved live configuration. Completed stages and committed successful cases are copied into the new run; failed, interrupted or unfinished cases are retried. Changed code, model identities, settings or selected input records require a separate run. Credentials can be replaced without changing the saved configuration because credential values are not recorded. Earlier runs are preserved. The fresh run manifest lists reused stages and per-stage reused case counts; `cases/` retains attempt history. Completed unscored judge observations are reused as unscored observations.

Ordinary `--input` without `--resume-incomplete` still reruns the requested target stage. Older runs without a case plan cannot recover calls that were never saved. Archived replay and paper scoring are unchanged.

A request left `IN_FLIGHT` after abrupt termination has an unknown remote outcome. Explicit continuation may send that request again. A response lost between the provider returning and the local atomic commit cannot be recovered, and provider-level exactly-once billing is not guaranteed. Responses committed locally are not requested again. This recovery operates between agent cases; a mutation case retains the original bounded internal generation/validation attempts.

## Missing data and download consent

The checkout includes checksum-verified scientific data bundles. Local working copies are used first, then bundled copies. With missing local records and bundles, use:

```bash
./ae.sh fetch-data
```

Verified local raw files are used to rebuild absent gzip bundles, using the frozen compression recipe and checking the exact committed gzip hash. Existing raw files or bundles with unexpected hashes are refused, not overwritten. The launcher verifies every manifest entry required by the Docker build (currently 21), including non-stage views and metadata, rather than checking only the master/reasoning/evaluation sentinels. Intact bundles are not unpacked on the host during this check. Docker still verifies raw hashes while unpacking.

The launcher runs `fetch-data --build-inputs` before building. `./ae.sh fetch-data` also restores raw working copies; both repair missing bundles from verified local bytes without network access or a consent prompt. If both copies are absent, the host-side recovery asks before accessing Hugging Face. Only an affirmative yes permits downloads from mlirbench/MLIRBench at revision 1af6c3c7a29946a7cd0e2aafd9386686d7fec9d1. Every file is verified against the committed manifest. Modified local files are refused rather than overwritten. --consent-download is the explicit noninteractive equivalent; do not pass it without reviewer consent. Historical runtime evidence absent from the published dataset cannot be invented by downloading it again.

If Python/zlib cannot recreate a frozen gzip hash, recovery fails clearly and retains the verified raw data. No replacement manifest or differently compressed distribution is accepted. Retry repair in the artifact's Python/Docker environment, or reuse an already-built image. No actual download is performed by the automated consent tests; their remote bytes are fixtures.

## Reuse an already-built image

```bash
# Recompute scoped metrics from a self-contained fresh checkpoint.
AE_SKIP_BUILD=1 ./ae.sh pipeline --stage metrics \
  --input /results/FRESH_COMPLETED_RUN --output /results/FRESH_METRICS_RUN
```

`AE_IMAGE_SOURCE=pull` is the default and downloads the versioned image for the Docker daemon’s CPU. Its source-fingerprint label must match the checkout. `AE_IMAGE_SOURCE=build` explicitly verifies the host bundles and builds one local image. `AE_IMAGE_SOURCE=local` uses a previously loaded image without pulling or building. Legacy explicit `AE_SKIP_BUILD=0` selects build and `AE_SKIP_BUILD=1` selects local mode when AE_IMAGE_SOURCE is unset. An explicit AE_IMAGE_SOURCE takes precedence. Live API opt-in is independent of image acquisition.

At runtime, fresh checkpoints supply their own program pairs and stage outputs. Fresh execution/reasoning/judging and fresh/mixed scoped metrics resolve only the published views actually needed. Archived execution checkpoints still require frozen master/history verification, and full archived paper reproduction still needs its published views. Source/parent checks and Hugging Face consent are not bypassed.


## CPU alternatives

ARM64 and Intel/AMD reviewers use the same commands. The launcher selects the Docker daemon's native architecture. Linux aarch64 is Docker arm64; Linux x86_64 is Docker amd64.

```bash
AE_PLATFORM=linux/amd64 ./ae.sh quickstart
AE_PLATFORM=linux/arm64 ./ae.sh quickstart
```

Selecting the opposite architecture uses emulation. The versioned publication workflow requires native ARM64 and AMD64 build-and-test runs, followed by fresh pulls and complete reviewer runs on both architectures. Per-version CI evidence identifies the tested source and image configurations. Both variants retain the same compiler/runtime pins and semantic compiler flags. CPU semantic checks are not performance benchmarks.

The distribution uses one registry reference, `ghcr.io/msspranavasai/mlirbench-ae:0.3.0rc4`. The registry manifest contains exactly the two tested native variants. Reviewers run the same command on both CPU types; they do not select separate agent images or start supporting services. See [container publication](docs/container-publication.md).

## Additional checks

```bash
./ae.sh test
./ae.sh check
./ae.sh reproduce-downstream --output /results/downstream-new
./ae.sh audit-provenance --all --workers 4 --output /results/author-label-audit-new
```

The historical label audit is an additional scientific diagnostic. It is no longer the main reviewer pipeline or a required rerun within quickstart. GPU checkpoint inference, training, judge-reliability evidence and backend experiments retain the limitations listed in STATUS.md. Archived replay supports the guidance in docs/woodruff-guidance.md; it does not certify every historical semantic label.

## One reviewer image and one container

The sole root Dockerfile builds all four agents, both dialect toolchains, the isolated original downstream compiler, archived records and optional live SDKs. One published version tag contains native AMD64 and ARM64 variants of that same environment. Docker selects the host-compatible variant; each invocation runs exactly one disposable reviewer container. There is no Compose stack, database or external compiler service.

The default launcher pulls the versioned registry reference and verifies that the selected image’s source fingerprint matches the checkout. Use AE_IMAGE to pin the published index digest when an immutable binary reference is desired. Version publication refuses to replace an existing tag with different platform digests. The default does not require manually loading architecture-specific tar files.

Keep the repository and image package private while preparing the candidate. GitHub repository visibility and GHCR package visibility must both be checked when the author releases it. Changing repository visibility alone must not be assumed to enable anonymous image pulls. See [the publication guide](docs/container-publication.md) for the public-download check.

For offline delivery, Docker can export/load the same image; AE_IMAGE_SOURCE=local prevents registry access. Older rc1 tar archives remain platform-specific and should be used with their matching source checkout.

Quickstart also writes `preservation.json`, comparing newly recalculated archived results and frozen inputs against the pre-hardening baseline. This check fails on changed numbers or records, without changing expected paper values, tolerances, labels or scoring rules. For a completed full quickstart run, repeat it with `./ae.sh verify-preservation --results /results/<run-folder>`. Scientific provenance and fresh experimental outcomes remain separate evidence.

## Compiler and RISC-V experiments

The original downstream IREE 3.12.0rc20260916 is installed at `/opt/paper-backend/bin`, isolated from the core execution compiler. Both are inside the same image. The backend wheels for ARM64 and AMD64 are pinned to exact official release URLs and SHA-256 values in requirements/backend.txt. Original compiler experiment sources and observations are preserved, and the portable driver resolves source text by its immutable published hash.

```bash
# Recompute 1,369 arithmetic checks from archived compiler observations.
./ae.sh compiler --output /results/compiler-replay
# Fresh transformations for all 857 Linalg or 845 Torch scan inputs.
./ae.sh compiler --mode fresh --dialect linalg --output /results/compiler-linalg
./ae.sh compiler --mode fresh --dialect torch --output /results/compiler-torch
# Fresh static RISC-V code generation for the 693 archived fusion-valid programs.
./ae.sh compiler --mode fresh --dialect linalg --riscv --workers 4 --output /results/compiler-riscv
```

Add `--limit 8` for a small demonstration. Subsets are explicitly identified and cannot certify the full experiment. Fresh commands save exact commands, IR, diagnostics, assembly and observed comparisons outside the repository. RISC-V analysis is cross-compilation/static instruction analysis, not measured RISC-V hardware speedup. A mismatch returns a failing status and retains both observed and archived values.

See [release scope](docs/release-scope.md) for the author-excluded judging ablation, the separately published dataset and remaining scientific checks. The code does not upload or replace the existing Zenodo dataset.

## Consolidated independent evidence

Quickstart readiness describes its own run. Later compiler experiments and external native CI evidence are assessed by `release-readiness`, using a checksum-pinned evidence index. See [release evidence](docs/release-evidence.md). The consolidated report credits complete verified experiments; it does not certify semantic labels or replace the original per-run reports.
