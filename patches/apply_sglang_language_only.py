#!/usr/bin/env python3
"""Allow --language-model-only for Qwen3_5ForConditionalGeneration (SGLang).

The vision tower of Qwen3.8-27B-FP8 is 0.92 GB on the GPU and this deployment serves text.
The base class already builds no tower and the loader skips unknown weights; only the
server allow-list misses the architecture. Without images MRoPE reduces to 1D RoPE, which
is what language-only mode uses.

Apply with the python of the SGLang env:
  /root/infer/sg22/bin/python patches/apply_sglang_language_only.py
"""
import importlib.util
from pathlib import Path
import sys

OLD = (
    "    LANGUAGE_MODEL_ONLY_ARCHITECTURES = (\n"
    '        "DeepseekV4ForCausalLM",\n'
)
NEW = (
    "    LANGUAGE_MODEL_ONLY_ARCHITECTURES = (\n"
    '        "Qwen3_5ForConditionalGeneration",\n'
    '        "DeepseekV4ForCausalLM",\n'
)


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        print("sglang not found in this python", file=sys.stderr)
        return 1
    path = Path(spec.origin).parent / "srt/server_args.py"
    text = path.read_text()
    if NEW in text:
        print(f"already patched: {path}")
        return 0
    if text.count(OLD) != 1:
        print(f"expected one match in {path}", file=sys.stderr)
        return 1
    path.write_text(text.replace(OLD, NEW))
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
