#!/usr/bin/env bash
set -euo pipefail
AE_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$AE_ROOT"
if [[ ${1:-} == fetch-data ]]; then
  shift
  exec python3 "$AE_ROOT/run.py" fetch-data "$@"
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable. Start Docker Desktop, then rerun." >&2
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
AE_IMAGE="${AE_IMAGE:-mlirbench-ae:reviewer}"
AE_OUTPUT_ROOT="${AE_OUTPUT_ROOT:-$AE_ROOT/../MLIRBench-AE-runs}"
mkdir -p "$AE_OUTPUT_ROOT"
AE_OUTPUT_ROOT="$(cd "$AE_OUTPUT_ROOT" && pwd)"
echo "Docker host: $AE_HOST_ARCH; image: $AE_PLATFORM ($AE_IMAGE)"
if [[ "$AE_PLATFORM" != "linux/$AE_HOST_ARCH" ]]; then
  echo "Cross-architecture execution requested; this run uses emulation."
fi
if [[ $# -eq 0 ]]; then set -- quickstart; fi
AE_NETWORK=none
AE_FLAGS=()
AE_HAS_OUTPUT=false
AE_ARGS=("$@")
for ((i=0;i<${#AE_ARGS[@]};i++)); do
  case "${AE_ARGS[$i]}" in
    --allow-paid-api)
      # This flag is the reviewer's explicit live-model opt-in. Default replay never enables networking.
      AE_NETWORK=bridge
      for AE_KEY in OPENAI_API_KEY OPENROUTER_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY GEMINI_API_KEY MLIRBENCH_MUTATION_MODEL OPENAI_MODEL MLIRBENCH_API_TIMEOUT_SECONDS MLIRBENCH_API_MAX_RETRIES MLIRBENCH_MUTATION_MAX_OUTPUT_TOKENS MLIRBENCH_REASONING_MAX_OUTPUT_TOKENS MLIRBENCH_JUDGE_MAX_OUTPUT_TOKENS MLIRBENCH_MUTATION_MAX_ATTEMPTS; do
        if [[ -n "${!AE_KEY:-}" ]]; then AE_FLAGS+=(--env "$AE_KEY"); fi
      done
      if [[ -n "${AE_ENV_FILE:-}" ]]; then
        [[ -f "$AE_ENV_FILE" ]] || { echo "AE_ENV_FILE does not exist" >&2; exit 1; }
        AE_FLAGS+=(--env-file "$AE_ENV_FILE")
      fi ;;
    --output|--output=*) AE_HAS_OUTPUT=true ;;
    --source)
      AE_SOURCE="${AE_ARGS[$((i+1))]}"
      [[ -f "$AE_SOURCE" ]] || { echo "Source file not found" >&2; exit 1; }
      AE_SOURCE="$(cd "$(dirname "$AE_SOURCE")" && pwd)/$(basename "$AE_SOURCE")"
      AE_FLAGS+=(-v "$AE_SOURCE:/input/program.mlir:ro")
      AE_ARGS[$((i+1))]=/input/program.mlir ;;
  esac
done
AE_RUN="${AE_ARGS[0]}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
if [[ "$AE_HAS_OUTPUT" == false && ( ${AE_ARGS[0]} == quickstart || ${AE_ARGS[0]} == pipeline || ${AE_ARGS[0]} == compiler ) ]]; then
  AE_ARGS+=(--output "/results/$AE_RUN")
fi
# Build-time requirements cover every frozen manifest entry, not just stage views.
case "${AE_SKIP_BUILD:-0}" in
  0)
    python3 "$AE_ROOT/run.py" fetch-data --build-inputs
    docker build --platform "$AE_PLATFORM" -t "$AE_IMAGE" . ;;
  1)
    if ! docker image inspect "$AE_IMAGE" >/dev/null 2>&1; then
      echo "AE_SKIP_BUILD=1 requires an existing local image: $AE_IMAGE. Build it first." >&2
      exit 1
    fi ;;
  *) echo "AE_SKIP_BUILD must be 0 or 1." >&2; exit 1 ;;
esac
AE_ACTUAL_PLATFORM="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$AE_IMAGE")"
if [[ "$AE_ACTUAL_PLATFORM" != "$AE_PLATFORM" ]]; then
  echo "Image platform $AE_ACTUAL_PLATFORM differs from $AE_PLATFORM. Rebuild for this platform (AE_SKIP_BUILD=0)." >&2
  exit 1
fi
docker run --rm --platform "$AE_PLATFORM" --network "$AE_NETWORK" \
  -e "MLIRBENCH_DOCKER_HOST_ARCH=$AE_HOST_ARCH" \
  -e "MLIRBENCH_DOCKER_PLATFORM=$AE_PLATFORM" \
  "${AE_FLAGS[@]}" -v "$AE_OUTPUT_ROOT:/results" "$AE_IMAGE" "${AE_ARGS[@]}"
if [[ "$AE_HAS_OUTPUT" == false && ( ${AE_ARGS[0]} == quickstart || ${AE_ARGS[0]} == pipeline || ${AE_ARGS[0]} == compiler ) ]]; then
  if [[ ${AE_ARGS[0]} == quickstart ]]; then AE_REPORT=START_HERE.md; elif [[ ${AE_ARGS[0]} == compiler ]]; then AE_REPORT=summary.json; else AE_REPORT=REPORT.md; fi
  echo "Read: $AE_OUTPUT_ROOT/$AE_RUN/$AE_REPORT"
fi
