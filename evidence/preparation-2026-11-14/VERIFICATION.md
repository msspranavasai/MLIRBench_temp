# Verified AE preparation workspace

The new MLIRBench-AE workspace builds from committed source 7f5c8837896e8326d8ef4a9473223b4ff98cf0c9. The exact ae.sh quickstart wrapper and an independent clean-Git-built image both completed the offline reviewer workflow.

- **31 core tests passed**, no skips or failures, in 28.009 seconds.
- Both real compiler/runtime smoke checks passed.
- **21 published distribution files and 288 downstream files verified**.
- All **76 downstream metric comparisons matched** and all **13 family-split checks passed**.
- DT1/DT2 predictions match original held-out identities and labels.
- The accepted arXiv v1 paper is frozen with a checksum and page/table references.
- Printed-value benchmark comparison: **23 match / 38 differ** under the stated aggregation.
- The clean-built image executed two real dataset diagnostic pairs on shared inputs.
- Eleven final LoRA adapters are located and remotely fingerprinted; weights are outside the CPU image.
- The original reference repository is clean; all **200** raw/processed files retain their hashes.
- Runtime used no network or provider credentials. No remote push, GPU training, or paid inference was performed.

**NOT_SUBMISSION_READY** is the intentional readiness result. See the copied readiness.json and START_HERE.md for unresolved scientific, experiment and hardware requirements. This result must not be replaced with a test-pass claim.

New generated runs belong in results/. These checked-in files preserve the author preparation baseline.
