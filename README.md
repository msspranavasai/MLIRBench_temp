# MLIRBench — CGO 2027 Artifact Evaluation

This is the **AE preparation workspace**. The sibling ../MLIRBench is the preserved reference repository.

Start with one command:

~~~bash
./ae.sh quickstart
~~~

It builds one Linux AMD64 image, verifies the data, executes compiler and core tests, and recomputes archived benchmark and downstream metrics **without API keys or runtime network access**.

Read the generated results/<run>/START_HERE.md first. It distinguishes working software from unresolved submission requirements. **The current artifact is not submission-ready.**

| Location | Purpose |
|---|---|
| src/mlirbench/ | Maintained offline execution, validation, scoring and CLI |
| experiments/ | Paper expectations, claim-to-command map and downstream inventory |
| data/ | Verified distribution bundles, historical evidence and provenance |
| tests/ | Unit tests and actual compiler/runtime integration |
| tools/data/ | Explicit data packaging and unpacking |
| optional/live_pipeline/ | Provider-based pipeline with separate dependencies |
| docs/ | Current evaluator instructions and limitations |
| archive/ | Preserved earlier documentation and verification |
| results/ | New generated outputs, ignored by Git |

Default reproduction covers archived model metrics, downstream saved-prediction metrics, and actual CPU compiler execution. GPU checkpoint inference and training remain a separately validated path.

Published inputs are immutable. Historical observation replay and new shared-input semantic auditing are labeled separately.

See [ARTIFACT.md](ARTIFACT.md) for commands, [STATUS.md](STATUS.md) for readiness gaps and [repository layout](docs/repository-layout.md) for the migration.
