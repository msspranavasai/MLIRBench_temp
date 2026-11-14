# Repository organization

Camera-Ready/MLIRBench-AE derives from Camera-Ready/MLIRBench commit 9af0239. The original repository and Active/MLIRBench remain reference sources. New work belongs here.

The maintained implementation is a Python package in src/mlirbench. CLI entry points are python run.py, python -m mlirbench after installation, and Docker. Compiler/runtime child processes import this same package.

The previous agents/graph/services live under optional/live_pipeline. Compatibility imports delegate execution to the maintained package. Provider requirements are separate from requirements/core.txt. Stale instructions and preparation tools moved to archive, retaining original content; their commands describe the previous layout.

Published data is restored from committed verified bundles into ignored working directories. The downstream snapshot contains original source, predictions and split files imported read-only from Nova, verified against remote SHA-256 hashes. Model weights, optimizer state, environments and credentials are excluded.

experiments/claims.json maps claims to commands and unfinished work. experiments/paper_reference.json holds independent printed expectations with manuscript provenance and rounding. Generated results never overwrite expectations or published inputs.

STATUS.md is the author-facing starting point. Quickstart creates START_HERE.md and machine-readable readiness evidence. Core test success and submission readiness are reported separately.
