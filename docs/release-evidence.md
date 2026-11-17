# Consolidated release evidence

Quickstart's `readiness.json` describes one run. It cannot discover later sibling compiler runs or native CI evidence by assuming they happened. `release-readiness` assesses explicitly linked independent evidence and writes a new report; it never edits old reports or scientific records.

## Pin and assess supplied evidence

The commands run with Python 3.10+ and the standard library, without Docker, GPUs or paid API calls. Run them from the matching code checkout. The local scientific bundles are verified and unpacked when assessment requires them; no replacement dataset or remote download is silently accepted.

```bash
python3 run.py index-evidence \
  --archive /absolute/path/ArtifactEvaluation.zip \
  --ci-evidence /absolute/path/ci-evidence \
  --output /absolute/path/release-review/evidence.json

python3 run.py release-readiness \
  --evidence /absolute/path/release-review/evidence.json \
  --output /absolute/path/release-review/assessment
```

The archive is the independent tester's unchanged results ZIP, including the tested source checkout, `MLIRBench-AE-runs/independent-.../STATUS.tsv`, stage outputs, quickstart and source-build reports, repeated samples, the full label audit and complete compiler program/record/assembly directories. If several independent runs are present, supply `--run independent-FOLDER_NAME` to `index-evidence`. TSV exit statuses locate the run; they are not accepted as proof of correctness.

Download CI artifacts from the workflow run for the **same tested source**, keeping their artifact-name directories. The optional `ci-evidence` directory must contain `image-evidence-amd64/amd64`, `image-evidence-arm64/arm64`, `pulled-image-evidence-amd64/amd64`, `pulled-image-evidence-arm64/arm64` and `published-image`. Each native artifact includes the original reports, not just a success flag. Without linked CI files, native platform coverage remains unverified.

`evidence.json` pins the archive and CI JSON hashes. Paths are relative to the index, so preserve that layout if transferring it. Hashes verify consistency, not authenticity or independent authorship: retain the tester's identity and original CI run links when distributing evidence. Do not update checksums merely to dismiss unexpected changes. Re-index only when deliberately selecting a new evidence set, preserving earlier indexes.

To run these commands inside the single reviewer container, put the archive, index and CI directories under the host `AE_OUTPUT_ROOT` mount, use their `/results/...` paths, and replace `python3 run.py` with `./ae.sh`. For example:

```bash
./ae.sh index-evidence --archive /results/evidence/ArtifactEvaluation.zip \
  --ci-evidence /results/evidence/ci-evidence \
  --output /results/evidence/evidence.json
./ae.sh release-readiness --evidence /results/evidence/evidence.json \
  --output /results/evidence/assessment
```

Use new output paths; existing indexes/reports are not overwritten. Indexing does not copy or extract the original ZIP. Assessment reads evidence without executing archived code or rerunning model/compilation experiments.

## Verification and interpretation

The assessor verifies:

- The source fingerprint calculated from the tester's actual archived source, matching their test records. Reporter and tested source fingerprints are displayed separately. Old tests are never attributed to changed reporting code.
- Native execution, complete non-skipped tests and every immutable preservation hash, including recalculated pipeline outputs, for the supplied quickstart and reported source-build run.
- Stage file hashes and record counts; repeated execution/sample outputs; paper agreement under the existing unchanged policy; downstream metrics independently recalculated from packaged predictions. Only machine-specific input path prefixes are normalized; metric values are not given a new tolerance.
- Exact complete compiler corpus IDs and input program hashes, compatible driver/toolchain/scan identity, command outcomes and operation counts recalculated from saved IR.
- All selected RISC-V cases, instruction counts/fingerprints recalculated from actual assembly, and comparison with archived outcomes. Expected compilation failures remain part of the comparison. This is static code generation, not hardware speedup measurement.
- Full audit coverage and its recorded program identities, labels and assessment counts. Audit completion remains separate from semantic label validity.
- Linked native AMD64 and ARM64 build/test and fresh registry-pull/test evidence, complete preservation coverage, paper agreement and published platform/source identities.

Inspect `assessment/REPORT.md` first. `assessment/readiness.json` lists gates, tested/reporter source identities, index/archive hashes and the hashes of inspected evidence members. Software, compiler, metrics, platform, scientific and build-provenance sections have separate statuses. Failed or incomplete functional evidence causes a nonzero command exit. A zero exit means the supplied functional evidence passed; it **does not** mean every scientific question is resolved.

For Syam's rc3 evidence the software, compiler, metrics and native-platform sections can pass. Published-label provenance remains `REVIEW_REQUIRED`; GPU inference and future apt snapshot freezing remain open. Judge-reliability stays author-excluded and unreproduced. Successful source installation does not establish a permanently frozen apt repository. None of these statuses changes paper values, labels, scoring, tolerances or claims.
