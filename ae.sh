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
# Default reviewer path pulls one versioned multi-platform image. Explicit
# legacy AE_SKIP_BUILD settings retain their local build/reuse meaning.
case "${AE_SKIP_BUILD:-unset}" in
  unset|0|1) ;;
  *) echo "AE_SKIP_BUILD must be 0 or 1." >&2; exit 1 ;;
esac
AE_IMAGE_SOURCE="${AE_IMAGE_SOURCE:-}"
if [[ -z "$AE_IMAGE_SOURCE" ]]; then
  case "${AE_SKIP_BUILD:-unset}" in
    unset) AE_IMAGE_SOURCE=pull ;;
    0) AE_IMAGE_SOURCE=build ;;
    1) AE_IMAGE_SOURCE=local ;;
  esac
fi
case "$AE_IMAGE_SOURCE" in
  pull) AE_DEFAULT_IMAGE=ghcr.io/msspranavasai/mlirbench-ae:0.3.0rc5 ;;
  build|local) AE_DEFAULT_IMAGE=mlirbench-ae:reviewer ;;
  *) echo "AE_IMAGE_SOURCE must be pull, build, or local." >&2; exit 1 ;;
esac
AE_IMAGE="${AE_IMAGE:-$AE_DEFAULT_IMAGE}"
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
AE_INSPECT_PLATFORM=()
if [[ "$(docker image inspect --help 2>/dev/null)" == *"--platform"* ]]; then
  AE_INSPECT_PLATFORM=(--platform "$AE_PLATFORM")
fi
case "$AE_IMAGE_SOURCE" in
  pull)
    docker pull --platform "$AE_PLATFORM" "$AE_IMAGE" ;;
  build)
    python3 "$AE_ROOT/run.py" fetch-data --build-inputs
    AE_SOURCE_ID="$(python3 "$AE_ROOT/tools/release/source_identity.py")"
    docker build --platform "$AE_PLATFORM" \
      --label "org.mlirbench.source-fingerprint=$AE_SOURCE_ID" -t "$AE_IMAGE" . ;;
  local)
    if ! docker image inspect ${AE_INSPECT_PLATFORM[@]+"${AE_INSPECT_PLATFORM[@]}"} "$AE_IMAGE" >/dev/null 2>&1; then
      echo "Local mode requires an existing image for $AE_PLATFORM: $AE_IMAGE. Use AE_IMAGE_SOURCE=build or pull." >&2
      exit 1
    fi ;;
esac
AE_ACTUAL_PLATFORM="$(docker image inspect ${AE_INSPECT_PLATFORM[@]+"${AE_INSPECT_PLATFORM[@]}"} --format '{{.Os}}/{{.Architecture}}' "$AE_IMAGE")"
if [[ "$AE_ACTUAL_PLATFORM" != "$AE_PLATFORM" ]]; then
  echo "Image platform $AE_ACTUAL_PLATFORM differs from $AE_PLATFORM. Rebuild with AE_IMAGE_SOURCE=build." >&2
  exit 1
fi
if [[ "$AE_IMAGE_SOURCE" == pull ]]; then
  AE_SOURCE_ID="$(python3 "$AE_ROOT/tools/release/source_identity.py")"
  AE_IMAGE_SOURCE_ID="$(docker image inspect ${AE_INSPECT_PLATFORM[@]+"${AE_INSPECT_PLATFORM[@]}"} --format '{{index .Config.Labels "org.mlirbench.source-fingerprint"}}' "$AE_IMAGE")"
  if [[ "$AE_IMAGE_SOURCE_ID" != "$AE_SOURCE_ID" ]]; then
    echo "Downloaded image does not match this checkout. Use the matching release or AE_IMAGE_SOURCE=build." >&2
    exit 1
  fi
fi
docker run --rm --platform "$AE_PLATFORM" --network "$AE_NETWORK" \
  -e "MLIRBENCH_DOCKER_HOST_ARCH=$AE_HOST_ARCH" \
  -e "MLIRBENCH_DOCKER_PLATFORM=$AE_PLATFORM" \
  ${AE_FLAGS[@]+"${AE_FLAGS[@]}"} -v "$AE_OUTPUT_ROOT:/results" "$AE_IMAGE" "${AE_ARGS[@]}"
if [[ "$AE_HAS_OUTPUT" == false && ( ${AE_ARGS[0]} == quickstart || ${AE_ARGS[0]} == pipeline || ${AE_ARGS[0]} == compiler ) ]]; then
  if [[ ${AE_ARGS[0]} == quickstart ]]; then AE_REPORT=START_HERE.md; elif [[ ${AE_ARGS[0]} == compiler ]]; then AE_REPORT=summary.json; else AE_REPORT=REPORT.md; fi
  echo "Read: $AE_OUTPUT_ROOT/$AE_RUN/$AE_REPORT"
fi
