# Downstream archived-prediction reproduction

Comparisons with archived final-figure CSVs: 76 match / 0 differ.
Family-split checks (DT1 + DT2): 13 pass / 0 overlap.
Invalid-label rows excluded under the original figure protocol: 0.

Metrics were recomputed from per-example predictions. Model inference and fine-tuning were not rerun.

| Task / method | n | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| Linalg fusion / Zero-shot | 693 | 31.89 | 26.80 |
| Linalg fusion / Few-shot mean | seed mean | 36.36 | 24.13 |
| Linalg fusion / SFT | 693 | 62.19 | 60.90 |
| Torch simplification / Zero-shot | 687 | 15.43 | 9.29 |
| Torch simplification / Few-shot mean | seed mean | 23.77 | 22.10 |
| Torch simplification / Balanced SFT | 687 | 58.08 | 47.96 |
| DT2 / Zero-shot | 344 | 57.85 | 48.51 |
| DT2 / 6-shot mean | 344 | 56.10 | 53.19 |
| DT2 / SFT | 344 | 84.30 | 80.87 |
