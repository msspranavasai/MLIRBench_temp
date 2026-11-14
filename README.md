# MLIRBench

CGO 2027 artifact work for *MLIRBench: A Benchmark and Dataset for Compiler and Hardware-Aware Reasoning*.

Build and inspect the published 1,768 records with one container:

```sh
docker build --platform linux/amd64 -t mlirbench:aec .
docker run --rm --platform linux/amd64 mlirbench:aec dataset-info
docker run --rm --platform linux/amd64 --network none mlirbench:aec check
docker run --rm --platform linux/amd64 --network none mlirbench:aec test
docker run --rm --platform linux/amd64 --network none mlirbench:aec score-recorded
```

Run a deterministic audit, saving results on the host:

```sh
mkdir -p results/aec
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD/results/aec:/results" mlirbench:aec \
  validate --id torch_seed4_prog40_T1_R1 --output /results/gather.jsonl
```

Use a new output filename for every run. `validate --sample` selects one record per available dialect/level/published-label stratum plus eight diagnostics (26 unique records in this snapshot). The default suite uses five seeds and two fresh-process repeats per program and input. A full sample takes several minutes under AMD64 emulation.

**Published data and labels are preserved.** Finite equal-output trials are reported as `NO_DIFFERENCE_OBSERVED`; they do not prove equivalence. Historical discrepancies, supported input domains, replay commands, and remaining artifact work are documented in [ARTIFACT.md](ARTIFACT.md) and [the audit](docs/REPRODUCIBILITY_AUDIT_2026-11-13.md).

Core inspection and validation require no credentials, network, remote validator, or model calls. The historical interactive model pipeline is available only through `python run.py pipeline`; its providers require separate credentials and may incur costs.
