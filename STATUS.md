# AE preparation status

The default reviewer route now follows mutation → execution → reasoning → judge → independently computed metrics, using archived model observations without paid API calls. Each stage can be replayed independently. Real CPU execution is available without provider credentials. Fresh model stages are optional source for additional experiments, outside the required reviewer route; no author-provided API keys, credits or access server are promised.

Run ./ae.sh quickstart and open its printed START_HERE.md. New run artifacts are stored outside this repository. Old preparation history and console logs have been moved to the sibling MLIRBench-AE-history folder; see HISTORY.md.

Published data, labels and paper reference values remain unchanged. The paper metric agreement policy is unchanged. Current results and per-run test outcomes are generated rather than embedded as asserted successes in this file.

The complete artifact is still under preparation. Remaining work includes:

- Original execution runtime inputs/toolchain provenance and scientific label conflicts.
- GPU checkpoint inference/access and evaluation on agreed hardware.
- Native AMD64 evaluator-host verification.
- Final dependency/build/resource freezing.
- Judge-reliability ablation: excluded from required reproduction by the author; not reproduced (docs/release-scope.md).
- Full compiler/RISC-V execution evidence; portable commands and the original isolated compiler are now included.

Paid provider calls were not used to validate this implementation. Tests use mocked model endpoints and real CPU execution where appropriate; saved subjective judge ratings are disclosed inputs to the no-cost paper replay.

Release hardening adds failed-mutation rejection (with explicit expected-invalid Level 5 handling), sanitized failure evidence, explicit OpenRouter Chat Completions routing, environment-file exclusions and a pre-hardening preservation guard. Quickstart uses the single `mlirbench-ae:reviewer` image and writes `preservation.json`; archived scientific claims still require the outstanding evidence above. Optional fresh validation differs from the paper’s historical fallback/clamping failure protocol, as disclosed in docs/live-configuration.md.
