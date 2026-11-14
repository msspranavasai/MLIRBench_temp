# MLIRBench: High-Level Architecture

This document describes the high-level architecture, directory layout, and core components of **MLIRBench**.

---

## 📁 Repository Structure

```text
MLIRBench/
│
├── agents/                      # Specialized agents driving the benchmark pipeline
│   ├── root_agent/              # Router and top-level orchestrator
│   ├── mutation_agent/          # Generates MLIR mutations
│   ├── execution_sandbox_agent/ # Validates, compiles, and executes MLIR programs
│   ├── llm_reasoning_agent/     # Generates semantic behavior predictions
│   └── judge_agent/             # Evaluates predictions against ground truth
│
├── graph/                       # Workflow lifecycle & routing logic
│   └── pipeline.py              # LangGraph workflow definition and state management
│
├── services/                    # External infrastructure integrations
│   └── validator_client.py      # Torch-MLIR validation client
│
├── docs/                        # System and API documentation
│
├── runs/                        # Historical outputs and pipeline artifacts
│
└── run.py                       # Interactive CLI entry point
```

# Core Components

## 1. Entry Point & Workflow Orchestration

- **`run.py`**
  The interactive command-line entry point for the benchmark.
  - **Responsibilities:** Accepts user parameters, initializes global pipeline state, triggers the LangGraph workflow, and persists the execution outputs.
- **`graph/`**
  The core execution engine of the benchmark framework.
  - **Responsibilities:** Manages the underlying pipeline state, coordinates agent-to-agent data flow, and enforces the deterministic LangGraph workflow execution.

---

## 2. Specialized Agents (`agents/`)

The framework relies on a multi-agent system where each agent is responsible for an isolated step of the evaluation lifecycle:

| Agent                       | Responsibility                                                                                         |
| :-------------------------- | :----------------------------------------------------------------------------------------------------- |
| **Root Agent**              | Acts as the primary router, dynamically directing execution requests through the workflow.             |
| **Mutation Agent**          | Programmatically generates syntactically valid and targeted MLIR mutations.                            |
| **Execution Sandbox Agent** | Provides an isolated environment to compile, validate, and execute MLIR code blocks.                   |
| **LLM Reasoning Agent**     | Instructs the target Large Language Model to produce semantic predictions about the MLIR code.         |
| **Judge Agent**             | Compares the LLM's semantic predictions against execution-derived ground truth data to score accuracy. |

---

## 3. Services & System Outputs

- **`services/`**
  Handles interfaces with external toolchains and native compilers. It currently encapsulates the **Torch-MLIR validation client** to assert syntax and semantic validity before execution.
- **`runs/`**
  The central cache and artifact repository for all historical pipeline executions.
  - **Generated Artifacts include:** Extracted mutations, validation/compiler warnings, execution stdout/stderr logs, and comparative evaluation metrics.

---

# Design Goals

MLIRBench is architected to prioritize five fundamental engineering principles:

- **Modular:** Every agent and service is decoupled, allowing components to be developed, optimized, or replaced in isolation.
- **Reproducible:** All execution artifacts, logs, and seeds are stored deterministically within the `runs/` directory to ensure predictable tracking.
- **Extensible:** Built on LangGraph, making it straightforward to inject new agents, custom validation steps, or alternative processing edges into the graph.
- **Dialect-Aware:** Built from the ground up to understand, target, and manipulate the explicit operation boundaries of different MLIR dialects.
- **Model-Agnostic:** Decoupled prompt engineering and evaluation states allow the benchmarking of any supported LLM backend without altering pipeline logic.
