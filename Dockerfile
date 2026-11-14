FROM --platform=linux/amd64 python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
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

COPY requirements/core.txt /tmp/requirements.txt

# Install the original pinned pipeline dependencies.
RUN python -m pip install --no-cache-dir \
    -r /tmp/requirements.txt

# Verify that the required compiler executables exist.
RUN python -c "import iree.compiler; import iree.runtime" && \
    command -v mlir-opt && \
    command -v torch-mlir-opt && \
    command -v iree-compile && \
    command -v iree-benchmark-module && \
    mlir-opt --version && \
    torch-mlir-opt --version && \
    iree-compile --version

# Maintained offline code and verified data only.
COPY pyproject.toml README.md ARTIFACT.md LICENSE run.py /app/
COPY src /app/src
COPY tools /app/tools
COPY tests /app/tests
COPY experiments /app/experiments
COPY docs /app/docs
COPY paper/reference /app/paper/reference
COPY data/bundle /app/data/bundle
RUN python tools/data/unpack_artifact_data.py
COPY data/downstream/snapshot.tar.gz data/downstream/manifest.json /app/data/downstream/
RUN python tools/data/unpack_downstream.py
COPY data/historical /app/data/historical
COPY data/provenance /app/data/provenance
RUN python -m pip install --no-cache-dir --no-deps . && test ! -f /app/.env
ENTRYPOINT ["python", "-m", "mlirbench"]
CMD ["--help"]
