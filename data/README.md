# Data and provenance

- bundle/ contains 21 deterministic compressed published JSON/JSONL/README files, verified against bundle/manifest.json.
- historical/ preserves original execution observations without relabeling.
- downstream/snapshot.tar.gz preserves 288 original downstream source/prediction/split files. Its manifest matches SHA-256 checksums computed on Nova.
- provenance/ records source commits, original published hashes, remote import hashes and final checkpoint identifiers.
- processed/ and downstream/records/ are generated working copies, ignored by Git.
- Model weights, optimizer state, credentials and virtual environments are excluded. Original references remain unchanged.

Docker unpacks and verifies both bundles automatically. For a local checkout, python run.py prepare-data restores working data. New outputs belong in results/.
