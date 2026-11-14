#!/usr/bin/env bash
set -euo pipefail
AE_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
AE_IMAGE=mlirbench-ae:dev
cd "$AE_ROOT"
if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable. Start Docker Desktop, then rerun this command." >&2
  exit 1
fi
docker build --platform linux/amd64 -t "$AE_IMAGE" .
mkdir -p "$AE_ROOT/results"
if [[ $# -eq 0 ]]; then set -- quickstart; fi
if [[ "$1" == "quickstart" && $# -eq 1 ]]; then
  AE_RUN="quickstart-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  docker run --rm --platform linux/amd64 --network none \
    -v "$AE_ROOT/results:/results" "$AE_IMAGE" quickstart --output "/results/$AE_RUN"
  echo "Read: $AE_ROOT/results/$AE_RUN/START_HERE.md"
else
  docker run --rm --platform linux/amd64 --network none \
    -v "$AE_ROOT/results:/results" "$AE_IMAGE" "$@"
fi
