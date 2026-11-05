#!/bin/bash
# Fine-tune Jamie's local model on this Mac, then install it for Jamie to use.
#
#   training/train.sh            # data -> train -> fuse -> GGUF -> install
#   training/train.sh --quick    # a 60-step trial run to check everything works
#
# Needs Apple silicon, about 25 GB of free disk, and Homebrew's llama.cpp
# (brew install llama.cpp). Run it from the repository root, inside Jamie's venv.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
work=${JAMIE_TRAIN_DIR:-$here/work}
train_model=${JAMIE_TRAIN_MODEL:-mlx-community/Qwen3.5-4B-4bit}   # trained in 4-bit (QLoRA)
base_model=${JAMIE_BASE_MODEL:-Qwen/Qwen3.5-4B}                   # fused in full precision
worlds=${JAMIE_TRAIN_WORLDS:-20}
iters=""
if [[ ${1:-} == --quick ]]; then
  worlds=2
  iters=60
fi
installed="$HOME/Library/Application Support/Jamie/models/jamie-4b.gguf"

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
  echo "Error: training uses MLX, which needs a Mac with Apple silicon." >&2
  exit 1
fi
for tool in llama-quantize llama-server; do
  if ! command -v "$tool" >/dev/null; then
    echo "Error: $tool is missing. Install llama.cpp: brew install llama.cpp" >&2
    exit 1
  fi
done
if ! python -c "import assistant" 2>/dev/null; then
  echo "Error: run this inside Jamie's virtual environment (source .venv/bin/activate)." >&2
  exit 1
fi
mkdir -p "$work"

echo "1/5  Building training data from Jamie's own tools ($worlds worlds)…"
python -m assistant.training.dataset --out "$work/data" --worlds "$worlds"

echo "2/5  Preparing the training tools (a separate environment, once)…"
if [[ ! -x $work/venv/bin/mlx_lm.lora ]]; then
  python3 -m venv "$work/venv"
  "$work/venv/bin/pip" install --quiet "mlx-lm==0.32.*" "transformers>=4.57" sentencepiece
fi

echo "3/5  Training a LoRA adapter on $train_model (the longest step; about 2-3 hours on"
echo "     an M2 with 16 GB; progress is printed every 20 steps)…"
"$work/venv/bin/mlx_lm.lora" --config "$here/lora.yaml" --model "$train_model" \
  --data "$work/data" --adapter-path "$work/adapter" ${iters:+--iters "$iters"}

echo "4/5  Merging the adapter into the original $base_model weights…"
rm -rf "$work/merged"
"$work/venv/bin/python" "$here/merge_lora.py" "$base_model" "$work/adapter" "$work/merged"

echo "5/5  Converting to GGUF for llama.cpp and quantizing to Q4_K_M…"
if [[ ! -d $work/llama.cpp ]]; then
  version=$(llama-server --version 2>&1 | sed -n 's/^version: \([0-9]*\).*/\1/p' | head -1)
  git clone --quiet --depth 1 ${version:+--branch "b$version"} \
    https://github.com/ggml-org/llama.cpp "$work/llama.cpp" ||
    git clone --quiet --depth 1 https://github.com/ggml-org/llama.cpp "$work/llama.cpp"
  "$work/venv/bin/pip" install --quiet -r \
    "$work/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
fi
"$work/venv/bin/python" "$work/llama.cpp/convert_hf_to_gguf.py" "$work/merged" \
  --outtype bf16 --outfile "$work/jamie-4b-bf16.gguf"
llama-quantize "$work/jamie-4b-bf16.gguf" "$work/jamie-4b-Q4_K_M.gguf" Q4_K_M

mkdir -p "$(dirname "$installed")"
if [[ -f $installed ]]; then
  mv "$installed" "${installed%.gguf}.previous.gguf"
fi
cp "$work/jamie-4b-Q4_K_M.gguf" "$installed"
echo
echo "Done. Jamie will use the fine-tuned model the next time the model server starts:"
echo "  $installed"
echo "To compare with the original, run the evaluations before and after, e.g.:"
echo "  jamie eval scenarios --runs 3"
echo "To go back to the original model, delete that file (the previous one is kept as"
echo "jamie-4b.previous.gguf)."
