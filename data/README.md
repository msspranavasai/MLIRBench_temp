# Published data

`bundle/` contains deterministic gzip copies of 21 published JSON/JSONL/README files, checked against `bundle/manifest.json`. `historical/` preserves original execution observations. `provenance/` records the hashes of all 200 original raw/processed files in the source workspace. The original workspace remains the archival reference.

Unpack once with `python tools/data/unpack_artifact_data.py`. Docker does this automatically. Working `processed/` files are ignored by Git. New experimental outputs belong in `results/`.
