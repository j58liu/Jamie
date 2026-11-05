"""Merge an mlx-lm LoRA adapter into the original Hugging Face weights.

mlx_lm.fuse saves the merged model in MLX's own layout (renamed tensors, moved conv
axes, shifted norm weights), which llama.cpp's converter does not read for Qwen3.5. This
adds each adapter's update to the matching original weight instead, and leaves every
other file untouched, so the result converts exactly like the published model:

    python training/merge_lora.py Qwen/Qwen3.5-4B training/work/adapter training/work/merged

The update is the one mlx-lm applies at run time: W + scale * (lora_b.T @ lora_a.T).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# MLX module paths -> Hugging Face tensor names (Qwen3.5 is stored as a vision-language
# checkpoint; MLX keeps only the language model).
PREFIXES = (
    ("language_model.model.", "model.language_model."),
    ("language_model.lm_head", "lm_head"),
    ("model.", "model."),
)


def hf_name(module: str) -> str:
    for mlx_prefix, hf_prefix in PREFIXES:
        if module.startswith(mlx_prefix):
            return hf_prefix + module[len(mlx_prefix) :] + ".weight"
    return module + ".weight"


def merge(base: Path, adapter: Path, out: Path) -> dict[str, int]:
    import mlx.core as mx

    config = json.loads((adapter / "adapter_config.json").read_text())
    scale = float(config["lora_parameters"]["scale"])
    weights = mx.load(str(adapter / "adapters.safetensors"))
    updates: dict[str, tuple] = {}
    for key in weights:
        if key.endswith(".lora_a"):
            module = key[: -len(".lora_a")]
            updates[hf_name(module)] = (weights[key], weights[f"{module}.lora_b"])
    if not updates:
        raise SystemExit("The adapter has no LoRA weights.")

    out.mkdir(parents=True, exist_ok=True)
    for item in base.iterdir():
        if item.suffix != ".safetensors" and item.is_file():
            shutil.copy2(item, out / item.name)
    merged = 0
    for shard in sorted(base.glob("*.safetensors")):
        tensors, metadata = mx.load(str(shard), return_metadata=True)
        for name, (lora_a, lora_b) in updates.items():
            if name in tensors:
                weight = tensors[name]
                delta = (scale * lora_b.T.astype(mx.float32)) @ lora_a.T.astype(mx.float32)
                if delta.shape != weight.shape:
                    raise SystemExit(f"{name}: update {delta.shape} != weight {weight.shape}")
                tensors[name] = (weight.astype(mx.float32) + delta).astype(weight.dtype)
                merged += 1
        mx.save_safetensors(str(out / shard.name), tensors, metadata={**metadata, "format": "pt"})
    missing = len(updates) - merged
    if missing:
        raise SystemExit(f"{missing} adapter weights had no matching tensor in {base}.")
    return {"merged": merged, "scale": int(scale)}


def resolve(model: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", help="Hugging Face repo or folder of the original model")
    parser.add_argument("adapter", type=Path)
    parser.add_argument("out", type=Path)
    arguments = parser.parse_args(argv)
    report = merge(resolve(arguments.base), arguments.adapter, arguments.out)
    print(f"Merged {report['merged']} weights into {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
