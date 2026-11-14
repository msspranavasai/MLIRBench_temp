# Start here: AE preparation status

**Organized for AE preparation; not submission-ready yet.**

Default evaluation uses archived model responses/judge observations, downstream saved predictions and actual CPU compilation/execution.

Established in this preparation work:

- Published input bundles are preserved and checksum-verified.
- Downstream source, predictions and original splits were located on Nova and copied read-only.
- All 76 downstream metric comparisons match archived final-figure CSVs.
- Family splits and DT2 held-out identities are audited from original files.
- Execution/validation is being retested after migration into the maintained package.
- Provider code and dependencies are optional.

Remaining author work:

1. Reconcile level-wise aggregation against the confirmed accepted paper; preserve current numerical differences for review.
2. Reconcile historical input/label provenance and state supported execution claims accurately.
3. Validate portable GPU/checkpoint evaluation and arrange distribution/access.
4. Complete native AMD64 and final dependency/resource verification.

Run ./ae.sh quickstart and read the generated results/<run>/START_HERE.md. It reports actual checks and open gates. A passing test suite does not substitute for those remaining requirements.
