# Artifact evaluation instructions

## Scope

The default route reproduces metrics from archived responses/judge observations and downstream predictions, without provider credentials or runtime network. This follows the email guidance summarized in docs/woodruff-guidance.md.

The artifact remains under preparation. STATUS.md and experiments/claims.json identify open requirements. No paper values or semantic labels are silently corrected.

## Start

Prerequisites: Docker, Linux AMD64 execution, and space for compiler tools and unpacked data. Apple Silicon can use emulation; native AMD64 remains the intended evaluator target. Building requires network access to pinned package sources. Runtime networking is disabled.

~~~bash
./ae.sh quickstart
~~~

This builds current source and creates a fresh result directory containing data verification, compiler execution, core tests, benchmark metrics, downstream metrics and a readiness report. Open the printed START_HERE.md first.

## Individual experiments

~~~bash
./ae.sh check
./ae.sh test
./ae.sh dataset-info
./ae.sh reproduce --output /results/paper-run
./ae.sh reproduce-downstream --output /results/downstream-run
./ae.sh validate --id torch_seed4_prog40_T1_R1 --output /results/pair-run.jsonl
./ae.sh readiness --results /results/quickstart-EXISTING-RUN
~~~

Use fresh output names. Outputs are JSON and, where appropriate, CSV/Markdown. Readiness exits 2 while requirements remain open. Quickstart exits nonzero for functional environment/test failure and separately reports submission readiness.

Direct container use:

~~~bash
docker build --platform linux/amd64 -t mlirbench-ae:dev .
docker run --rm --platform linux/amd64 --network none mlirbench-ae:dev dataset-info
~~~

## What is recomputed

The reproduce command derives dataset counts, model summaries, level/dialect summaries and confusion counts from immutable records. Subjective judge observations remain archived inputs; objective label comparisons and aggregations are recalculated. Denominators and missing observations are explicit. Independent manuscript expectations live in experiments/paper_reference.json. They reference the author-confirmed accepted arXiv v1 PDF, preserved with a checksum and page/table citations.

The reproduce-downstream command independently computes accuracy, macro-F1, per-class F1, dialect metrics and false-EQUIV rates from per-example predictions. It compares them with original final-figure CSVs, audits family separation and verifies DT2 held-out IDs/gold labels. Metric serialization agreement is checked at 12 decimal places, much finer than printed paper percentages. This does not change exact program-output comparison.

The validate command is a separately versioned shared-input audit: five seeds, two fresh-process repeats per program, positional input mapping and operation-aware preconditions. Incompatible signatures and invalid/unsupported inputs do not become divergences. Finite equal outputs do not prove equivalence.

## GPU and live providers

Original downstream source and adapter metadata are indexed under experiments/downstream. Checkpoint inference, training, weight distribution and reviewer GPU access still need validation. Archived-prediction replay is the implemented CPU route.

Live API generation lives under optional/live_pipeline, with separate requirements and explicit invocation. It is outside the default image and quickstart. Authors arrange any needed paid API access.

See docs/known-limitations.md for scientific caveats and docs/repository-layout.md for source organization. Runtime measurements from the new image are recorded in preparation evidence; CPU metric replay is not presented as GPU training reproduction.
