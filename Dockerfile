FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

ARG TARGETARCH
ENV MLIRBENCH_IMAGE_ARCH=${TARGETARCH}

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
COPY requirements/live.txt /tmp/live-requirements.txt
COPY requirements/runtime-lock.txt /tmp/runtime-lock.txt

# Install the same pinned compiler/runtime versions for the selected architecture.
RUN python -m pip install --no-cache-dir \
    -r /tmp/requirements.txt -r /tmp/live-requirements.txt -c /tmp/runtime-lock.txt

# Verify that the required compiler executables exist.
RUN python -c "import iree.compiler; import iree.runtime" && \
    command -v mlir-opt && \
    command -v torch-mlir-opt && \
    command -v iree-compile && \
    command -v iree-benchmark-module && \
    mlir-opt --version && \
    torch-mlir-opt --version && \
    iree-compile --version

# Preserve the original downstream compiler in an isolated environment. Core
# validation continues to use the pinned 2024 toolchain in /usr/local/bin.
COPY requirements/backend.txt /tmp/backend-requirements.txt
RUN python -m venv --system-site-packages /opt/paper-backend && \
    /opt/paper-backend/bin/python -m pip install --no-cache-dir --no-deps -r /tmp/backend-requirements.txt && \
    /opt/paper-backend/bin/iree-compile --version

# Maintained reviewer workflow, original agent implementations and verified data.
COPY Dockerfile ae.sh pyproject.toml README.md ARTIFACT.md STATUS.md LICENSE DATA_LICENSE.md CITATION.cff run.py .gitignore /app/
COPY requirements /app/requirements
COPY optional/live_pipeline /app/optional/live_pipeline
COPY src /app/src
COPY tools /app/tools
COPY tests /app/tests
COPY experiments /app/experiments
COPY docs /app/docs
COPY paper/reference /app/paper/reference
COPY data/README.md /app/data/README.md
COPY data/bundle /app/data/bundle
RUN python tools/data/unpack_artifact_data.py
COPY data/downstream/snapshot.tar.gz data/downstream/manifest.json /app/data/downstream/
RUN python tools/data/unpack_downstream.py
COPY data/historical /app/data/historical
COPY data/provenance /app/data/provenance
RUN python -m pip install --no-cache-dir --no-deps . && test ! -f /app/.env
ENTRYPOINT ["python", "-m", "mlirbench"]
CMD ["--help"]
