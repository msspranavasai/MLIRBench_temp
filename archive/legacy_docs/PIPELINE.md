# MLIRBench Pipeline Architecture

This document describes the end-to-end multi-agent execution workflow of MLIRBench, managed via LangGraph.

---

## Workflow Overview

```text
[User Inputs] -> Root Agent -> Mutation Agent -> Validation -> Execution Sandbox -> LLM Reasoning Agent -> Judge Agent -> Final Output
```

### Pipeline Inputs

The pipeline requires three user-provided parameters:

- **MLIR Program:** The target compiler IR source code.
- **Dialect:** `Linalg` or `Torch`.
- **Mutation Level:** Severity tier from `L1` to `L5`.

---

### Core Components and Stages

#### 1. Root Agent

Serves as the workflow entry point. It validates the user configuration (program syntax, chosen dialect, and mutation level), initializes the centralized pipeline state payload, and routes the transaction downstream.

#### 2. Mutation Agent

Analyzes the input code structure and programmatically applies transformations based on the designated target level:

| Level  | Mutation Type    | Description & Examples                                                                | Expected Execution Result              |
| :----- | :--------------- | :------------------------------------------------------------------------------------ | :------------------------------------- |
| **L1** | Syntactic        | Semantics-preserving surface changes (e.g., attribute reordering, layout formatting). | Semantically Equivalent (`EQUIV`)      |
| **L2** | Parameter        | Modifies operation parameters (e.g., constants, shape modifications).                 | Equivalent or Divergent                |
| **L3** | Operator         | Replaces operations with semantically related alternatives.                           | Equivalent or Divergent                |
| **L4** | Structural       | Structurally alters control-flow, shape reorganization, or block movement.            | Equivalent or Divergent                |
| **L5** | Failure-Inducing | Intentionally injects type mismatches, shape violations, or invalid ops.              | Compilation or Runtime Error (`ERROR`) |

#### 3. Validation

Before execution, mutations must clear static verifier tooling configured per dialect to ensure structural validity:

- **Linalg Dialect:** Verified locally using the native compiler toolchain:
  ```bash
  mlir-opt
  ```
- **Torch Dialect:** Forwarded to a remote validation service via a REST endpoint:

```
POST /validate
```

#### 4. Execution Sandbox Agent

Establishes baseline ground-truth behavior by executing both the original and mutated programs in an isolated runtime environment using IREE.

```plaintext
MLIR Source -> IREE Compiler -> VMFB Artifact -> IREE Runtime -> Execution Output
```

**Performance Profiling:** Hardware execution latency metrics (mean, standard deviation, percentiles) are captured via:

```bash
iree-benchmark-module
```

**Ground Truth Evaluation:** Outputs are categorized into one of three execution states:

- **EQUIV:** Program outputs match identically.
- **DIVERGE:** Programs complete successfully but yield different data results.
- **ERROR:** Compilation or execution failed.

---

#### 5. LLM Reasoning Agent

Prompts target language models to statically analyze the programs and predict runtime outcomes without executing the code. Supported model providers include OpenAI, Anthropic, Gemini, and OpenRouter.

**Responsibilities:** Predicts semantic equivalence, maps system failure behaviors, and provides text explanations of code deviations.

---

#### 6. Judge Agent

Compares the LLM Reasoning Agent's static predictions against the absolute physical results obtained by the Execution Sandbox.

**Responsibilities:** Scores prediction correctness, evaluates natural language explanation quality, assigns final benchmark performance labels (EQUIV, DIVERGE, ERROR), and exports the execution records.
