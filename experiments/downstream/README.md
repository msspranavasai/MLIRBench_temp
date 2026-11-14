# Downstream reproduction

Run:

~~~bash
./ae.sh reproduce-downstream --output /results/downstream-run
~~~

The CPU command recomputes final-figure metrics from per-example predictions and checks original family splits. It does not load an LLM or regenerate predictions.

The verified snapshot contains DT1 Linalg/Torch fold predictions and fold JSON; DT2 zero-shot, six-shot and SFT predictions with original splits; independent final-figure CSVs; and original training/evaluation source and adapter configurations. Remote and local SHA-256 hashes agree.

Adapters are indexed from Nova and excluded from the default CPU image. DT2 uses Qwen/Qwen2.5-Coder-14B-Instruct; its adapter is about 1.1 GB. The original notebook records an A100 80 GB. GPU inference must be made portable, tested and coordinated before being claimed as reproduced.
