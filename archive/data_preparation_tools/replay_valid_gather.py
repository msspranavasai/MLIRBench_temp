"""Targeted diagnostic; writes exact values and outputs without changing data."""
import json
import numpy as np
from agents.execution_sandbox_agent.tools.execution import compile_mlir, run_compiled, compare_outputs, _compile_extra_args
from agents.execution_sandbox_agent.tools.pair_ground_truth import _argument_names

rows = [json.loads(l) for l in open("data/processed/MLIRBench/master/master.jsonl")]
r = next(r for r in rows if r["program_id"] == "torch_seed4_prog40_T1_R1")
compiled = [compile_mlir(r[k], dialect="torch") for k in ("input_mlir", "mutated_mlir")]
report = {"program_id": r["program_id"], "published": r["judge__outputs_diverge"], "compiler_flags": _compile_extra_args("llvm-cpu"), "trials": []}
for seed in (42, 123):
    values = [np.random.default_rng(seed).standard_normal((6,3)).astype(np.float32), np.array([0,1,2,0,1,2], dtype=np.int64)]
    outputs = []
    for field, c in zip(("input_mlir", "mutated_mlir"), compiled):
        outputs.append([run_compiled(c["compiled_module"], mlir_source=r[field], input_values=dict(zip(_argument_names(r[field]), values)), timeout_seconds=60) for _ in range(2)] if c["status"] == "success" else [c])
    report["trials"].append({"seed": seed, "inputs": [{"dtype": str(v.dtype), "shape": list(v.shape), "data": v.tolist()} for v in values], "runs": outputs, "comparison": compare_outputs(outputs[0][0].get("output"), outputs[1][0].get("output"))})
print(json.dumps(report, indent=2))
