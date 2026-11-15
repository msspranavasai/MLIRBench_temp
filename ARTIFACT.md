# Artifact evaluation instructions

## Scope

The default route reproduces metrics from archived responses/judge observations and downstream predictions, without provider credentials or runtime network. This follows the email guidance summarized in docs/woodruff-guidance.md.

The artifact remains under preparation. STATUS.md and experiments/claims.json identify open requirements. No paper values or semantic labels are silently corrected.

## Start

Prerequisites: Docker on an ARM64 or AMD64 host, and space for compiler tools and unpacked data. The launcher selects the Docker daemon's architecture, including when using a remote Docker context. Apple Silicon uses native Linux ARM64; Intel/AMD hosts use Linux AMD64. Both platforms use the same pinned compiler/runtime versions. Building requires network access to pinned package sources. Runtime networking is disabled.

~~~bash
./ae.sh quickstart
~~~

This builds current source and creates a fresh result directory containing data verification, compiler execution, core tests, benchmark metrics, downstream metrics and a readiness report. Open the printed START_HERE.md first.

## Reviewer CPU alternatives

An ARM device is not required. On an Intel/AMD Docker host, ./ae.sh quickstart automatically builds linux/amd64. On Apple Silicon or an ARM64 Docker host, the same command automatically builds linux/arm64. Linux reports these as x86_64 and aarch64 respectively; Docker calls them amd64 and arm64. Windows reviewers need Docker configured for Linux containers and a Bash environment such as WSL2.

Explicit alternatives use the same launcher:

~~~bash
# Intel/AMD variant (native on an x86-64 Docker host).
AE_PLATFORM=linux/amd64 ./ae.sh quickstart

# ARM64 variant (native on an ARM64 Docker host).
AE_PLATFORM=linux/arm64 ./ae.sh quickstart
~~~

Running the opposite architecture requires emulation. The launcher and environment.json identify this. Both variants passed 35 core tests and all 76 downstream comparisons on the preparation Mac; only ARM64 was native. A native Intel/AMD evaluator-host run remains required before marking that hardware gate verified.

## Combined reviewer image

Authors can build both variants under one local tag using a manifest-capable image store, such as Docker Desktop's containerd image store:

~~~bash
docker buildx build --platform linux/amd64,linux/arm64 --load \
  --tag mlirbench-ae:reviewer .
~~~

This creates a local multi-platform image. It does not upload or publish an image. Docker automatically selects its ARM64 variant on ARM64 hosts and its AMD64 variant on Intel/AMD hosts. Engines without multi-platform local image support can use the single-platform ./ae.sh route above. An eventual registry/archive distribution must retain both platform manifests.

From the repository root, run the already-built combined image offline:

~~~bash
mkdir -p results
docker run --rm --network none \
  -e "MLIRBENCH_DOCKER_HOST_ARCH=$(docker info --format '{{.Architecture}}')" \
  -v "$PWD/results:/results" \
  mlirbench-ae:reviewer quickstart --output /results/reviewer-run
~~~

Use a fresh output directory name for each run. The host-architecture environment value allows the native-execution check to verify the selected variant. With the combined image, omit --platform to let Docker select the native variant. These are CPU evaluation paths; GPU checkpoint inference remains a separate requirement.

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

Images are tagged mlirbench-ae:dev-arm64 or mlirbench-ae:dev-amd64 so the two architectures remain distinguishable. The environment report records the Docker host architecture, container architecture, emulation status and visible CPU counts. Runs through ./ae.sh supply the host metadata needed to verify native execution. A direct Docker run without that metadata reports native execution as unverified.

To explicitly test emulation or build for another supported platform:

~~~bash
AE_PLATFORM=linux/amd64 ./ae.sh quickstart
~~~

The launcher reports when this differs from the Docker host. AMD64 names the x86-64 instruction set used by both Intel and AMD; it is not a processor speed or quality rating. The native platform avoids instruction emulation, but actual performance also depends on CPU, memory and Docker resource allocation. The launcher adds no CPU limit; Docker Desktop's VM allocation still applies. Canonical semantic audits retain their existing deterministic compiler flags and do not become optimized performance measurements.

Direct container use (ARM64 example; replace arm64 with amd64 on an Intel/AMD Docker host):

~~~bash
docker build --platform linux/arm64 -t mlirbench-ae:dev-arm64 .
docker run --rm --platform linux/arm64 --network none mlirbench-ae:dev-arm64 dataset-info
~~~

## What is recomputed

The reproduce command derives dataset counts, model summaries, level/dialect summaries and confusion counts from immutable records. Subjective judge observations remain archived inputs; objective label comparisons and aggregations are recalculated. Denominators and missing observations are explicit. Independent manuscript expectations live in experiments/paper_reference.json. They reference the author-confirmed accepted arXiv v1 PDF, preserved with a checksum and page/table citations.

The reproduce-downstream command independently computes accuracy, macro-F1, per-class F1, dialect metrics and false-EQUIV rates from per-example predictions. It compares them with original final-figure CSVs, audits family separation and verifies DT2 held-out IDs/gold labels. Metric serialization agreement is checked at 12 decimal places, much finer than printed paper percentages. This does not change exact program-output comparison.

The validate command is a separately versioned shared-input audit: five seeds, two fresh-process repeats per program, positional input mapping and operation-aware preconditions. Incompatible signatures and invalid/unsupported inputs do not become divergences. Finite equal outputs do not prove equivalence.

## GPU and live providers

Original downstream source and adapter metadata are indexed under experiments/downstream. Checkpoint inference, training, weight distribution and reviewer GPU access still need validation. Archived-prediction replay is the implemented CPU route.

Live API generation lives under optional/live_pipeline, with separate requirements and explicit invocation. It is outside the default image and quickstart. Authors arrange any needed paid API access.

See docs/known-limitations.md for scientific caveats and docs/repository-layout.md for source organization. Runtime measurements from the new image are recorded in preparation evidence; CPU metric replay is not presented as GPU training reproduction.
