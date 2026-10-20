#!/bin/bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: scripts/start-local-model.sh [PATH_TO_GGUF]' \
    '' \
    'Start the local Qwen model at http://127.0.0.1:8080 without internet access.' \
    'With no path, use Qwen3.5-2B Q4_K_M already stored in the llama.cpp cache.' \
    'Alternatively, supply the path to an existing local GGUF file.' \
    '' \
    'Install llama.cpp and download the model once using the README setup steps.' \
    'Optional settings: JAMIE_CONTEXT_SIZE, JAMIE_CACHE_RAM_MIB, JAMIE_MODEL_REPO.' \
    'This launcher does not download model files. Press Ctrl+C to stop the server.'
}

if [[ $# -eq 1 && ( $1 == --help || $1 == -h ) ]]; then
  usage
  exit 0
fi

context_size=${JAMIE_CONTEXT_SIZE:-32768}
cache_ram_mib=${JAMIE_CACHE_RAM_MIB:-512}
model_repo=${JAMIE_MODEL_REPO:-unsloth/Qwen3.5-2B-GGUF:Q4_K_M}

if [[ ! $context_size =~ ^[1-9][0-9]{3,5}$ ]] || \
  (( context_size < 2048 || context_size > 131072 )); then
  printf '%s\n' 'Error: JAMIE_CONTEXT_SIZE must be an integer from 2048 through 131072.' >&2
  exit 2
fi
if [[ ! $cache_ram_mib =~ ^[0-9]{1,4}$ ]]; then
  printf '%s\n' 'Error: JAMIE_CACHE_RAM_MIB must be an integer from 0 through 8192.' >&2
  exit 2
fi
cache_ram_mib=$((10#$cache_ram_mib))
if (( cache_ram_mib > 8192 )); then
  printf '%s\n' 'Error: JAMIE_CACHE_RAM_MIB must be an integer from 0 through 8192.' >&2
  exit 2
fi

if [[ $# -gt 1 || ( $# -eq 1 && $1 == -* ) ]]; then
  usage >&2
  exit 2
fi

if ! command -v llama-server >/dev/null 2>&1; then
  printf '%s\n' 'Error: llama-server is not installed or is not on PATH.' \
    'Install llama.cpp once with: brew install llama.cpp' >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  if [[ ! -f $1 || ! -r $1 ]]; then
    printf 'Error: model file does not exist or is not readable: %s\n' "$1" >&2
    exit 1
  fi
  model_args=(--model "$1")
else
  model_args=(-hf "$model_repo")
  printf '%s\n' \
    "Starting the cached $model_repo model offline." \
    'If the cache is missing, follow the first-download steps in README.md or provide a local GGUF path.' >&2
fi

exec llama-server \
  "${model_args[@]}" --offline --no-mmproj \
  --host 127.0.0.1 --port 8080 \
  --gpu-layers all --flash-attn on \
  --ctx-size "$context_size" --parallel 1 --cache-ram "$cache_ram_mib" \
  --jinja --reasoning-format deepseek \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --perf
