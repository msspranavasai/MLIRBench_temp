FROM --platform=linux/amd64 python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLIR_OPT=mlir-opt \
    TORCH_MLIR_MODE=local \
    TORCH_MLIR_OPT=torch-mlir-opt \
    MLIR_VALIDATOR_TIMEOUT=60

WORKDIR /app

# Native dependencies and LLVM/MLIR tools.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgomp1 \
        llvm \
        mlir-19-tools \
    && rm -rf /var/lib/apt/lists/*

# Expose mlir-opt using its versioned Debian executable.
RUN MLIR_BIN="$(find /usr/bin -maxdepth 1 \
        -name 'mlir-opt-*' | sort -V | tail -n 1)" && \
    test -n "$MLIR_BIN" && \
    ln -s "$MLIR_BIN" /usr/local/bin/mlir-opt

COPY requirements.txt /tmp/requirements.txt

# Install the original pinned pipeline dependencies.
RUN python -m pip install --no-cache-dir \
    -r /tmp/requirements.txt

# Install the standalone Torch-MLIR compiler.
RUN python -m pip install --no-cache-dir \
    torch-mlir==20261101

# Verify that the required compiler executables exist.
RUN python -c "import iree.compiler; import iree.runtime" && \
    command -v mlir-opt && \
    command -v torch-mlir-opt && \
    command -v iree-compile && \
    command -v iree-benchmark-module && \
    mlir-opt --version && \
    torch-mlir-opt --version && \
    iree-compile --version

# Explicit allowlist prevents credentials, local environments, and archives entering the image.
COPY pyproject.toml README.md ARTIFACT.md LICENSE artifact.py recorded_scores.py run.py /app/
COPY src /app/src
COPY agents /app/agents
COPY graph /app/graph
COPY services /app/services
COPY scripts /app/scripts
COPY tests /app/tests
COPY docs /app/docs
COPY data/artifact_bundle /app/data/artifact_bundle
RUN python scripts/unpack_artifact_data.py
COPY data/replay /app/data/replay
COPY data/manifests /app/data/manifests
RUN python -m pip install --no-cache-dir --no-deps . && test ! -f /app/.env

ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
