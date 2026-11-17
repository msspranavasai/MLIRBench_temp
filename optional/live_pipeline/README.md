# Original model agents

The maintained reviewer CLI in src/mlirbench/pipeline.py invokes these original agents through lazy adapters when a fresh stage is explicitly requested. They are included in the single Docker image. Default replay uses archived outputs and makes no model calls.

Use ./ae.sh pipeline from the repository root. See ARTIFACT.md for stages, consent, credentials and new test cases. run_pipeline.py is a compatibility wrapper for the maintained CLI. The old graph is retained as implementation reference; reviewer stage checkpoints are managed by the maintained runner.

See [fresh API configuration](../../docs/live-configuration.md) for model identities, credentials and request limits shared by the reviewer preflight and actual agent clients.

This directory is optional source for additional experiments. The planned AE route replays archived responses and judge observations and requires no API keys or credits. No author-provided API credentials, credits or remote API service are included. Runtime credential/consent guards remain enabled for users who independently choose to run live code. Invalid live judge responses are rejected and preserved as failed cases; the full mutation rule and recorded inspection metadata are passed to judging.
