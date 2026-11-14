# Artifact readiness

**NOT_SUBMISSION_READY**

| Check | Status |
|---|---|
| published_data | PASS |
| compiler_execution | PASS |
| automated_tests | PASS |
| native_cpu_execution | PASS |
| paper_metric_agreement | REVIEW_REQUIRED |
| archived_downstream_replay | PASS |
| archived_model_metrics | OPEN |
| execution_label_provenance | OPEN |
| gpu_checkpoint_evaluation | OPEN |
| native_amd64 | OPEN |
| dependency_freeze | OPEN |
| judge_reliability | OPEN |
| compiler_backend_experiments | OPEN |

## Remaining actions

- paper_metric_agreement: Inspect the associated comparison or evidence file.
- archived_model_metrics: Reconcile and document Table 4 averaging against the confirmed accepted paper; preserve the raw pooled and alternative aggregation results.
- execution_label_provenance: Historical shared-input discrepancies and unsupported domains require a scientific resolution.
- gpu_checkpoint_evaluation: Arrange checkpoint access, port inference and validate agreed GPU hardware.
- native_amd64: The default launcher now uses native ARM64 on Apple Silicon. Native AMD64 evaluator verification remains outstanding; ARM64 success does not establish AMD64 native performance.
- dependency_freeze: Freeze apt/toolchain provenance and validate a fresh evaluator setup.
- judge_reliability: Locate and package the frozen 100-pair/900-response judging study, saved scores and aggregation script; rerunning paid judging is optional.
- compiler_backend_experiments: Port and test the original compiler/RISC-V experiment drivers. Source and archived scan records are present; new code generation has not been reproduced.
