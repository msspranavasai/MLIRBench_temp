# Archived-results reproduction

Paper reference: **FINAL_ACCEPTED_MANUSCRIPT_CONFIRMED**.
Checks: **23 match / 38 differ**.

Metric aggregation uses archived responses and judge observations; it does not validate historical semantic labels.

| Model | Available / expected | Label accuracy (%) | Recorded aggregate | Result /10 | Explanation /4 |
|---|---:|---:|---:|---:|---:|
| claude-haiku-4-5 | 1768 / 1768 | 69.34 | 57.44 | 3.54 | 2.70 |
| claude-sonnet-4-5 | 1768 / 1768 | 77.38 | 62.95 | 4.10 | 2.82 |
| gemini-3.1-pro-preview | 1763 / 1768 | 92.65 | 74.18 | 5.04 | 3.17 |
| glm-5.1 | 1768 / 1768 | 72.85 | 54.93 | 3.59 | 2.24 |
| gpt-4.1-mini | 1768 / 1768 | 67.14 | 51.79 | 3.13 | 2.28 |
| gpt-oss-120b | 1768 / 1768 | 78.22 | 60.38 | 3.98 | 2.53 |
| kimi-k2.6 | 1768 / 1768 | 78.51 | 58.31 | 3.79 | 2.34 |
| llama-4 | 1768 / 1768 | 66.86 | 45.47 | 2.53 | 1.77 |
| qwen3-coder-next | 1768 / 1768 | 68.38 | 49.50 | 2.96 | 2.02 |

## Reference differences

- tab:level_results claude-sonnet-4-5 L1: expected 63.82; computed 63.85.
- tab:level_results claude-sonnet-4-5 L3: expected 65.38; computed 65.58.
- tab:level_results claude-sonnet-4-5 L4: expected 65.13; computed 65.21.
- tab:level_results claude-sonnet-4-5 L5: expected 59.11; computed 58.77.
- tab:level_results gpt-oss-120b L1: expected 60.90; computed 60.79.
- tab:level_results gpt-oss-120b L2: expected 61.08; computed 60.98.
- tab:level_results gpt-oss-120b L3: expected 65.02; computed 65.05.
- tab:level_results gpt-oss-120b L4: expected 62.47; computed 62.54.
- tab:level_results gpt-oss-120b L5: expected 52.59; computed 52.30.
- tab:level_results kimi-k2.6 L1: expected 57.48; computed 57.44.
- tab:level_results kimi-k2.6 L2: expected 56.52; computed 56.43.
- tab:level_results kimi-k2.6 L3: expected 63.62; computed 63.71.
- tab:level_results kimi-k2.6 L5: expected 51.67; computed 51.33.
- tab:level_results claude-haiku-4-5 L1: expected 59.14; computed 59.17.
- tab:level_results claude-haiku-4-5 L2: expected 56.70; computed 56.85.
- tab:level_results claude-haiku-4-5 L3: expected 56.29; computed 56.53.
- tab:level_results claude-haiku-4-5 L4: expected 60.54; computed 60.63.
- tab:level_results claude-haiku-4-5 L5: expected 53.57; computed 53.37.
- tab:level_results glm-5.1 L1: expected 59.34; computed 59.37.
- tab:level_results glm-5.1 L2: expected 58.40; computed 58.43.
- tab:level_results glm-5.1 L3: expected 61.50; computed 61.60.
- tab:level_results glm-5.1 L4: expected 61.69; computed 61.84.
- tab:level_results glm-5.1 L5: expected 32.78; computed 32.50.
- tab:level_results gpt-4.1-mini L1: expected 61.23; computed 61.32.
- tab:level_results gpt-4.1-mini L2: expected 55.90; computed 56.07.
- tab:level_results gpt-4.1-mini L3: expected 53.46; computed 53.60.
- tab:level_results gpt-4.1-mini L4: expected 59.82; computed 59.95.
- tab:level_results gpt-4.1-mini L5: expected 26.88; computed 26.41.
- tab:level_results qwen3-coder-next L1: expected 56.62; computed 56.67.
- tab:level_results qwen3-coder-next L2: expected 52.21; computed 52.18.
- tab:level_results qwen3-coder-next L3: expected 58.23; computed 57.93.
- tab:level_results qwen3-coder-next L4: expected 57.95; computed 58.10.
- tab:level_results qwen3-coder-next L5: expected 21.18; computed 21.08.
- tab:level_results llama-4 L1: expected 52.42; computed 52.50.
- tab:level_results llama-4 L2: expected 49.36; computed 49.40.
- tab:level_results llama-4 L3: expected 52.54; computed 52.44.
- tab:level_results llama-4 L4: expected 53.63; computed 53.77.
- tab:level_results llama-4 L5: expected 18.15; computed 17.88.
