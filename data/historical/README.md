# Preserved execution evidence

historical_execution.jsonl is the extracted provenance sidecar for the 1,768 published pairs. execution_agent_unified_rows.csv.gz is an exact compressed copy of the original read-only Supabase CSV export, including all 1,804 source rows. source_manifest.json records the archive, raw-source, sidecar and published-master hashes. The audit independently checks every published row and source program against those records.

Embedded inputs are historical metadata. This export does not preserve actual per-program runtime arguments or establish the original compiler version. No historical label is changed by packaging these files. See docs/execution-provenance.md and the generated audit report for interpretation.
