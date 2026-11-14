# Evaluation scope agreed by email

Prof. Jackson Woodruff's response to the November 13, 2026 email supports:

1. Reproducing metrics from archived model responses instead of requiring paid API inference.
2. Author-arranged credentials/access if live API reproduction is included.
3. An executable compiler environment that reviewers use to reproduce results; installation alone is insufficient.
4. Checkpoints plus possible author-hosted SSH access or reviewers with comparable GPUs. Checkpoint-only sufficiency and hardware allocation are not guaranteed.
5. Experiments ideally finishing in less than a couple of days, with fast retry cycles. No hard runtime limit was set; the review period is three and a half weeks.

Default evaluation follows the archived-response CPU route. Archived judge observations are inputs to aggregation; subjective judgments are not regenerated. Downstream metrics are recomputed from per-example predictions. Checkpoint inference and training are separate hardware-dependent paths.

This guidance does not certify the artifact, validate historical labels or resolve manuscript/data differences.
