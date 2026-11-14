# Execution Infrastructure

This document describes how MLIRBench validates, executes, and evaluates generated mutations.

The execution layer is responsible for establishing ground truth by comparing the behavior of an original MLIR program against a mutated version.

---

# Overview

MLIRBench uses execution-based verification to determine whether a mutation:

- Preserves program behavior
- Changes program behavior
- Introduces compilation or runtime failures

Execution is performed on both the original and mutated programs using IREE.

---

# Execution Workflow

```text
Original Program
        │
        ├── Validation
        │
        ├── Compilation
        │
        └── Execution
                │
                ▼

Mutated Program
        │
        ├── Validation
        │
        ├── Compilation
        │
        └── Execution
                │
                ▼

Output Comparison
        │
        ▼

Ground-Truth Label
```

---

# Validation

Before execution, generated mutations must pass validation.

## Linalg Dialect

Validated locally using:

```bash
mlir-opt --verify-diagnostics
```

## Torch Dialect

Validated remotely through the Torch-MLIR validation service.

Internally, the service executes:

```bash
torch-mlir-opt --verify-diagnostics
```

Programs that fail validation are labeled with error and are not executed further.

---

# Compilation and Execution

Validated programs are compiled and executed using IREE.

Execution flow:

```text
  MLIR
    ↓
IREE Compiler
    ↓
  VMFB
    ↓
IREE Runtime
    ↓
 Output
```

MLIRBench executes both:

- Original program
- Mutated program

using identical inputs.

---

# Input Generation

Runtime inputs are automatically generated from the program signature.

Supported input types include:

- Scalars
- Vectors
- Tensors
- Multi-dimensional tensors

The same generated inputs are reused across both programs to ensure a fair comparison.

---

# Output Comparison

After execution, outputs from both programs are compared.

Possible outcomes:

| Label   | Description                                  |
| ------- | -------------------------------------------- |
| EQUIV   | Outputs match                                |
| DIVERGE | Outputs differ                               |
| ERROR   | Validation, compilation, or execution failed |

Output comparison currently relies on exact output matching.

---

# Performance Benchmarking

In addition to correctness evaluation, MLIRBench records execution performance.

Benchmarking is performed using:

```bash
iree-benchmark-module
```

Collected metrics may include:

- Mean latency
- Standard deviation
- Minimum latency
- Median latency
- P90 latency
- P99 latency

These measurements are stored as benchmark metadata and are not used for semantic labeling.

---

# Produced Artifacts

The execution layer produces:

- Validation status
- Compilation status
- Execution status
- Program outputs
- Benchmark measurements
- Error logs
- Ground-truth labels

These artifacts are passed to downstream agents for reasoning and evaluation.
