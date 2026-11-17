# Scientific and execution limitations

Published labels and paper results are preserved. New audits do not silently replace them.

- The disputed historical Linalg outputs reproduce using different independently embedded inputs. That is historical observation replay, not a semantic counterexample.
- Stored gather metadata contains invalid indices. Canonical validation rejects it and generates bounded indices for supported operations.
- Changed signatures are incomparable. Unsupported domains, nonfinite outputs, errors, timeouts and nondeterminism receive distinct outcomes.
- The core audit reports matching finite trials as NO_DIFFERENCE_OBSERVED; the reviewer pipeline maps this to EQUIV explicitly scoped to the tested valid-input suite, without claiming universal equivalence. Program-output comparison has no numeric tolerance. Signed zeros compare equal; byte fingerprints separately detect instability.
- Linalg helper/dynamic-domain coverage remains incomplete. The diagnostic sample is not a dataset error-rate estimate.
- The accepted arXiv v1 paper is confirmed. Table 4 numerical agreement uses the declared author-defined tolerance of 0.15 points out of 100 and requires unchanged rankings/level orderings. Exact historical aggregation and formatting remain incompletely reconstructed; numerical agreement does not resolve semantic label provenance or constitute a CGO badge decision. Exact differences are preserved in JSON.
- The author reports that the older Level 2 table used 149 Linalg pairs before removal of seed3_linalg_prog27_L2_R2; the artifact independently confirms 148 current Linalg pairs, 160 Torch pairs, and absence of that ID. This is a likely explanation for eight older-model Level 2 differences. The excluded pair's eight original judge scores have not been recovered, so exact numerical attribution remains unconfirmed. Reports preserve the differences and include experiments/level2_dataset_change.json as an attributed provenance note. GPT-OSS L4 remains a separate rounding discrepancy.
- Subjective model scores average available judge rows; label accuracy includes all attempted pairs. Gemini has five missing judge rows. Denominators are emitted.
- The original downstream figure script filters invalid labels. Replay reports every exclusion and readiness cannot pass that gate silently.
- Saved-prediction replay is distinct from inference or training. Final adapter weights have been copied to the supplemental reviewer assets; base-model weights and portable GPU execution still require validation.
- Original GPU scripts retain historical absolute paths. They are provenance, not advertised portable commands.
- The launcher pulls one versioned image for the Docker host's native ARM64 or AMD64 architecture by default; local source building is explicit. Publication requires full native build/test and fresh-pull verification on both CPUs, with evidence retained per version. The Debian apt/build snapshot remains unfrozen. Explicit cross-architecture runs use emulation and are identified in environment evidence. CPU timings across architectures are not directly interchangeable.

- Compiler/RISC-V source and scan records are archived, but fresh backend code generation is a separate unfinished reproduction path.

- Input policy valid-domain-v5 adds sign/zero coverage with operation-domain-checked fallback. The adapter no longer promotes a partial successful trial from an unverified suite. Source verification covers the exact original 1,804-row Supabase CSV and all 1,768 extracted published records. Actual historical runtime inputs remain unrecorded. The new audit reports current witnesses and structural conflicts without changing published labels; see docs/execution-provenance.md.
