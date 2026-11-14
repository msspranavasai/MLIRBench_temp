#!/usr/bin/env bash
set -euo pipefail
AE_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$AE_ROOT"
if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable. Start Docker Desktop, then rerun this command." >&2
  exit 1
fi
AE_HOST_ARCH="$(docker info --format '{{.Architecture}}')"
case "$AE_HOST_ARCH" in
  aarch64|arm64) AE_HOST_ARCH=arm64 ;;
  x86_64|amd64) AE_HOST_ARCH=amd64 ;;
  *) echo "Unsupported Docker host architecture: $AE_HOST_ARCH" >&2; exit 1 ;;
esac
AE_PLATFORM="${AE_PLATFORM:-linux/$AE_HOST_ARCH}"
case "$AE_PLATFORM" in
  linux/arm64|linux/amd64) ;;
  *) echo "AE_PLATFORM must be linux/arm64 or linux/amd64." >&2; exit 1 ;;
esac
AE_IMAGE="${AE_IMAGE:-mlirbench-ae:dev-${AE_PLATFORM#linux/}}"
echo "Docker host: $AE_HOST_ARCH; image: $AE_PLATFORM ($AE_IMAGE)"
if [[ "$AE_PLATFORM" != "linux/$AE_HOST_ARCH" ]]; then
  echo "Cross-architecture execution requested; this run uses emulation."
fi
docker build --platform "$AE_PLATFORM" -t "$AE_IMAGE" .
mkdir -p "$AE_ROOT/results"
if [[ $# -eq 0 ]]; then set -- quickstart; fi
if [[ "$1" == "quickstart" && $# -eq 1 ]]; then
  AE_RUN="quickstart-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  docker run --rm --platform "$AE_PLATFORM" --network none \
    -e "MLIRBENCH_DOCKER_HOST_ARCH=$AE_HOST_ARCH" \
    -e "MLIRBENCH_DOCKER_PLATFORM=$AE_PLATFORM" \
    -v "$AE_ROOT/results:/results" "$AE_IMAGE" quickstart --output "/results/$AE_RUN"
  echo "Read: $AE_ROOT/results/$AE_RUN/START_HERE.md"
else
  docker run --rm --platform "$AE_PLATFORM" --network none \
    -e "MLIRBENCH_DOCKER_HOST_ARCH=$AE_HOST_ARCH" \
    -e "MLIRBENCH_DOCKER_PLATFORM=$AE_PLATFORM" \
    -v "$AE_ROOT/results:/results" "$AE_IMAGE" "$@"
fi
