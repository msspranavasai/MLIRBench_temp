# Installation Guide

This document provides environment-specific installation, dependency management, and verification procedures for MLIRBench.

---

## Supported Environments

### 1. Local Development Environment

The core benchmarking framework, agent state machines, and evaluation logic are optimized for Apple Silicon macOS architectures. The following reference systems are used for local pipeline development, debugging, and dataset inspection:

- MacBook Pro 13" (Apple M1)
- MacBook Pro 14" (Apple M3)
- MacBook Pro 16" (Apple M4 Max)
- MacBook Pro 15" (Apple M5)

### 2. Remote Benchmark Environment

A cloud-hosted environment deployed on a **Google Cloud Compute Engine VM**. It acts as the canonical validation backend, running the remote validation service to expose `torch-mlir-opt` for verifying Torch dialect mutations.

---

## Dependency Installation

### 1. LLVM & Native MLIR Toolchain

Install the upstream LLVM distribution locally using Homebrew to obtain the native MLIR optimization tools:

```bash
brew install llvm
```

#### Environment Paths

The typical installation path on Apple Silicon is:

```plaintext
/opt/homebrew/opt/llvm/bin/mlir-opt
```

Export this binary location to your active environment variables:

```bash
export MLIR_OPT="/opt/homebrew/opt/llvm/bin/mlir-opt"
```

Verify your local installation:

```bash
$MLIR_OPT --version
```

### 2. IREE Compilation & Runtime Engine

MLIRBench targets IREE for lowered compilation, hardware execution, and profile benchmarking. Install the required modules via Python's package manager:

```bash
pip install iree-compiler iree-runtime
```

Verify the module installations run without runtime errors:

```bash
python -c "import iree.compiler"
python -c "import iree.runtime"
```

---

## Remote Validation Service Setup

To minimize local configuration footprints, Torch dialect mutations are handled through a remote validation API. Local development environments do not require a native `torch-mlir` build.

### 1. Link the Validation Backend

Set the target API URL mapping to the Google Cloud validation host:

```bash
export TORCH_MLIR_API="http://<validator-host>:8000"
```

### 2. Verify Service Availability

Query the endpoint health status via `curl`:

```bash
curl http://<validator-host>:8000/health
```

#### Expected Response Payload

```json
{
  "status": "ok",
  "tool": "torch-mlir-opt",
  "available": true
}
```

---

## Verification Checklist

Run this comprehensive diagnostic block before launching any pipeline execution to guarantee your environment is configured correctly:

```bash
# 1. Verify Local MLIR Compiler CLI
$MLIR_OPT --version

# 2. Verify Local IREE Components
python -c "import iree.compiler; import iree.runtime; print('IREE: OK')"

# 3. Verify Remote Torch-MLIR Network Bridge
curl -fsS "$TORCH_MLIR_API/health"
```

All three telemetry checks must yield zero exit errors before proceeding to benchmark generation or evaluation loops.

---

## Next Steps

Once your environment checklist is entirely clear, proceed to the following operational blueprints:

- `PIPELINE.md` — In-depth breakdown of multi-agent graph mutations.
- `EXECUTION.md` — Sandboxed compilation profiles and execution matrix parameters.
- `REPRODUCIBILITY.md` — Isolation policies for regression checking and results synchronization.

To launch the interactive generation pipeline directly:

```bash
python run.py
```
