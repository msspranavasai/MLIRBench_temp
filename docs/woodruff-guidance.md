# Evaluation guidance and submission scope

Prof. Jackson Woodruff's response to the November 13, 2026 email supports:

1. Reproducing metrics from archived model responses instead of requiring paid API inference.
2. Author-arranged access if live API experiments are considered necessary for evaluation; a remote access machine is suggested as a way to provide it. The email does not require paid live reruns when the archived-response route is used.
3. An executable compiler environment that reviewers use to reproduce results; installation alone is insufficient.
4. Checkpoints plus possible author-hosted SSH access or reviewers with comparable GPUs. Checkpoint-only sufficiency and hardware allocation are not guaranteed.
5. Experiments ideally finishing in less than a couple of days, with fast retry cycles. No hard runtime limit was set; the review period is three and a half weeks.

Default evaluation follows the archived-response CPU route. Archived judge observations are inputs to aggregation; subjective judgments are not regenerated. Downstream metrics are recomputed from per-example predictions. Checkpoint inference and training are separate hardware-dependent paths.

This guidance does not certify the artifact, validate historical labels or resolve manuscript/data differences.

## Submission plan

The planned reviewer route uses immutable archived model responses and judge observations, independent recomputation of objective correctness/score arithmetic/paper aggregates, downstream saved-prediction replay, and actual compiler/CPU execution. Reviewers run those commands themselves; installation or viewing precomputed totals alone is not the intended evaluation. Archived responses are not presented as freshly generated outputs. Scientific provenance and the hardware-dependent paper experiments retain their existing open issues.

Live agents remain available as optional source for additional experiments, separate from the required reproduction instructions. No API keys, credits or remote API access server are being supplied or promised, and evaluators are not expected to purchase access for the required route. The code still checks consent, models, credentials and budgets before live calls for anyone who independently chooses that path. Developer configuration is documented separately from the reviewer quickstart.

If a live paid experiment later becomes necessary to support an AE claim, its access and funding must be arranged with the evaluators at that point; optional source alone does not reproduce that experiment. This plan does not remove GPU/checkpoint or compiler experiment obligations, and the email is guidance rather than an artifact acceptance or badge decision.
