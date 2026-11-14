# Optional live model pipeline

This preserves the provider-driven construction/reasoning/judging pipeline. It is outside the default offline AE path and may incur API costs. Install `requirements.txt` here separately and invoke `python optional/live_pipeline/run_pipeline.py` only when live generation is intended. Its execution imports delegate to the tested `mlirbench.execution` package.

No credentials are copied from the reference workspace. Evaluator reproduction uses archived model responses and stored judge records by default. This path is not exercised by the default core tests.
