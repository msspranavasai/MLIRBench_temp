# One image, one reviewer container

Published image reference for this candidate:

```text
ghcr.io/msspranavasai/mlirbench-ae:0.3.0rc2
```

The single root Dockerfile packages the four agents, local validation tools, both IREE toolchains, archived observations and metric computation. The registry index has two native variants of that same environment: linux/amd64 and linux/arm64. Docker chooses the CPU-compatible variant; the reviewer runs one container per invocation. There is no supporting database, remote compiler server, Compose stack or per-agent image.

## Reviewer command

From the matching source checkout:

```bash
./ae.sh quickstart
```

The launcher downloads the versioned image for the Docker daemon's CPU and checks its source-fingerprint label against this checkout. The experiment container runs with networking disabled unless the user explicitly requests an optional paid API experiment. No provider API credentials, credits or Nova access are needed for the default archived route. Image downloading is separate from experiment networking.

While the GitHub repository and GHCR package are private, registry authentication and read access are required. Authenticate to GHCR using GitHub's documented package-access mechanism; never put credentials in the source, image or reports. Once the author makes the package public, public reviewers can pull without registry credentials.

Use `AE_IMAGE_SOURCE=build ./ae.sh quickstart` to build the same Dockerfile locally. Use `AE_IMAGE_SOURCE=local AE_IMAGE=mlirbench-ae:reviewer ./ae.sh quickstart` for an already loaded local image without registry access. An explicit AE_PLATFORM selects a platform; choosing another CPU architecture requires emulation and is identified in the output. AE_IMAGE may be set to the published index digest for a binary-immutable reference.

## Publication verification

The versioned workflow:

1. Builds AMD64 and ARM64 on native GitHub-hosted runners.
2. Runs the full offline workflow, tests, numerical agreement and preservation checks on each.
3. Checks private-repository visibility before pushing any source-bearing layers.
4. Pushes each tested platform by digest, with no public architecture-specific tags. The pushed configuration must match the tested one.
5. Assembles exactly those two manifests under one version tag. A conflicting existing version tag is not overwritten.
6. Pulls that one tag on new native AMD64 and ARM64 runners and repeats the complete reviewer workflow.

The OCI source/revision/version labels and per-platform CI evidence identify the build. The workflow publishes an index digest in the `published-image` artifact. A successful container publication validates this packaged software workflow; it does not resolve the separately disclosed historical label, GPU or incomplete RISC-V scientific checks.

## When the author is ready to make it public

The temporary GitHub repository and GHCR package are currently private. Keep both private during preparation.

When releasing, check both the repository's visibility and the `mlirbench-ae` package's visibility. GitHub packages have their own visibility settings; do not assume changing the repository alone makes its image anonymously downloadable. Set the container package to Public when public image distribution is intended, and check that repository/package permissions still match the intended audience.

Then run the **Verify public image downloads** workflow. It uses fresh native runners and a fresh Docker configuration with no registry login. Passing on both platforms establishes that an anonymous reviewer can pull the one versioned image and execute the documented default workflow. No image rebuild is required solely to change visibility.

References:

- https://docs.docker.com/build/building/multi-platform/
- https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry
- https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility
