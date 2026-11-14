# Reproducibility Guide

This document describes the infrastructure and guidelines used to ensure consistent, reproducible MLIRBench experiments.

MLIRBench leverages a centralized architecture to guarantee that all experimental runs share identical validation behavior and compiler toolchain versions.

---

## Reference Environment

The canonical validation backend for Torch dialect mutations is deployed with the following software stack:

- **Cloud Infrastructure:** Google Cloud Compute Engine VM
- **Operating System:** Ubuntu 22.04 LTS
- **Runtime Environment:** Python 3.10
- **Core Frameworks:** Torch-MLIR, FastAPI, Uvicorn

---

## Infrastructure Design

To eliminate local environment drift, Torch dialect mutations are validated remotely on a centralized server rather than using local binaries.

```text
[ Local Pipeline ]
        │
        ▼
[ Validator Client ]
        │
   (HTTP POST)
        ▼
[ Google Cloud VM ]
        │
        ▼
[ torch-mlir-opt ]
        │
        ▼
[ Validation Result ]
```

This decoupled design ensures that every evaluation run across different client machines utilizes the exact same `Torch-MLIR` compiler build.

---

## Validator Service Configuration

The remote VM hosts a lightweight FastAPI service wrapping the native MLIR verification tool:

```bash
torch-mlir-opt --verify-diagnostics
```

### Network Integration

The service is exposed on port `8000`. Connect your local pipeline execution by configuring the environment variable:

```bash
export TORCH_MLIR_API="http://<validator-host>:8000"
```

---

## Step-by-Step Reproduction Procedure

Follow this sequence to execute a reproducible benchmark run:

1. **Verify Connectivity:** Ensure the remote validator service is live by querying its health status.
2. **Export Environment:** Set the canonical API endpoint variable in your active shell.
3. **Trigger Pipeline:** Execute the interactive entry point script:

```bash
python run.py
```

4. **Collect Artifacts:** Verify that the generated outputs have been compiled and structured inside the local repository:

```plaintext
runs/
```

---

## Strict Reproducibility Guidelines

To maintain experiment integrity and validity across independent trials, adhere to the following rules:

- **Compiler Lock:** Always pin and verify the `Torch-MLIR` wheel/commit version running on the remote validator.
- **Network Isolation:** Do not switch validator hosts mid-suite; keep configuration metadata uniform across comparative runs.
- **Artifact Preservation:** Always preserve raw generated mutations alongside their corresponding hardware execution logs.
- **Telemetry Metadata:** Always log the base hardware state, pipeline parameters, and seed values inside your final result packages.
