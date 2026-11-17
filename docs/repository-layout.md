# Repository layout

MLIRBench-AE is the maintained AE workspace. The sibling MLIRBench and the older Active/MLIRBench remain preserved references.

| Location | Purpose |
|---|---|
| src/mlirbench/pipeline.py | Ordered stage replay/fresh orchestration and score arithmetic |
| src/mlirbench/stage_artifacts.py | Checkpoint hashes, parent validation and resume |
| src/mlirbench/dataset.py | Local-first data resolution and consented pinned downloads |
| src/mlirbench/live.py | Lazy adapters to the original model agents |
| src/mlirbench/execution/ | Real compiler/runtime, input domains and historical diagnostics |
| src/mlirbench/evaluation/ | Independent score aggregation and downstream metrics |
| optional/live_pipeline/agents/ | Original mutation, execution, reasoning and judge implementations |
| data/ | Frozen scientific inputs, prediction/score observations and provenance |
| experiments/ | Independent manuscript expectations, numerical policy and claim map |
| tests/ | Boundary, consent, scoring and actual execution tests |
| requirements/ | Core toolchain and live-agent dependency pins |
| docs/ | Current instructions and scientific limitations |
| paper/reference/ | Accepted manuscript and source identification |

The Docker image includes both offline and optional live agent code. No model client is initialized during archived replay. Expected paper values are loaded only after metrics are calculated.

Old runs, build logs, preparation evidence and legacy documentation were moved outside the repository into MLIRBench-AE-history. HISTORY.md records their location. New reviewer artifacts go to MLIRBench-AE-runs by default. Scientific input records remain under data; these include necessary original outputs and validation evidence, not disposable console logs.

`experiments/release_invariants.json` records pre-hardening hashes of frozen scientific inputs and recalculated archived outputs. `src/mlirbench/preservation.py` checks them after quickstart; mismatches fail the workflow. It never supplies metric values to the scorer. New live outputs are independent and are not required to match archived output hashes. A preservation PASS does not establish the correctness of historical labels or unresolved paper claims.
