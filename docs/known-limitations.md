# Scientific and execution limitations

Published labels and paper results are preserved. New audits do not silently replace them.

- The disputed historical Linalg outputs reproduce using different independently embedded inputs. That is historical observation replay, not a semantic counterexample.
- Stored gather metadata contains invalid indices. Canonical validation rejects it and generates bounded indices for supported operations.
- Changed signatures are incomparable. Unsupported domains, nonfinite outputs, errors, timeouts and nondeterminism receive distinct outcomes.
- Equal finite trials mean NO_DIFFERENCE_OBSERVED, not equivalence. Program-output comparison has no numeric tolerance. Signed zeros compare equal; byte fingerprints separately detect instability.
- Linalg helper/dynamic-domain coverage remains incomplete. The diagnostic sample is not a dataset error-rate estimate.
- The accepted arXiv v1 paper is confirmed. Its level-table averaging still needs reconciliation. A numerical reference difference is not automatically a dataset defect.
- Subjective model scores average available judge rows; label accuracy includes all attempted pairs. Gemini has five missing judge rows. Denominators are emitted.
- The original downstream figure script filters invalid labels. Replay reports every exclusion and readiness cannot pass that gate silently.
- Saved-prediction replay is distinct from inference or training. Checkpoint weights remain on the author cluster; portable GPU execution and reviewer access need validation.
- Original GPU scripts retain historical absolute paths. They are provenance, not advertised portable commands.
- The launcher builds for the Docker host's native ARM64 or AMD64 architecture by default. Native ARM64 is tested locally on Apple Silicon; native AMD64 verification and a frozen apt/build snapshot remain outstanding. Explicit cross-architecture runs use emulation and are identified in environment evidence. CPU timings across architectures are not directly interchangeable.

- Compiler/RISC-V source and scan records are archived, but fresh backend code generation is a separate unfinished reproduction path.
