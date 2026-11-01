#!/usr/bin/env python3
"""Make Qwen3.5 MTP defer embed_tokens/lm_head allocation to vLLM's existing
target-model sharing pass in v1/spec_decode/eagle.py.

Apply inside the vLLM venv:
  /home/jovyan/infer/venv017/bin/python patches/apply_mtp_share_patch.py
"""
from pathlib import Path
import sys

TARGET = Path(
    "/home/jovyan/infer/venv017/lib/python3.12/site-packages/vllm/"
    "model_executor/models/qwen3_5_mtp.py"
)

EMBED_OLD = """        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )
"""

EMBED_NEW = """        # MTP shares target model embeddings; allocation happens in eagle.py.
        self.embed_tokens = None
"""

LM_HEAD_OLD = """        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
"""

LM_HEAD_NEW = """        # MTP shares target model lm_head; allocation happens in eagle.py.
        self.lm_head = None
"""

REMAP_OLD = """        def remap_weight_names(weights):
            for name, weight in weights:
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                elif any(key in name for key in ["embed_tokens", "lm_head"]):
                    if "embed_tokens" in name:
                        name = name.replace("language_model.", "")
                else:
                    continue
                yield name, weight
"""

REMAP_NEW = """        def remap_weight_names(weights):
            for name, weight in weights:
                # MTP shares target embeddings/lm_head; do not load duplicates.
                if any(key in name for key in ("embed_tokens", "lm_head")):
                    continue
                if not name.startswith("mtp."):
                    continue
                yield name.replace("mtp.", "model."), weight
"""


def main() -> int:
    if not TARGET.exists():
        print(f"missing: {TARGET}")
        return 1
    original = TARGET.read_text()
    if "MTP shares target model embeddings; allocation happens in eagle.py." in original:
        print(f"already patched: {TARGET}")
        return 0

    backup = TARGET.with_suffix(TARGET.suffix + ".bak")
    if not backup.exists():
        backup.write_text(original)

    text = original
    for label, old, new in (
        ("embed_tokens", EMBED_OLD, EMBED_NEW),
        ("lm_head", LM_HEAD_OLD, LM_HEAD_NEW),
        ("load_weights remap", REMAP_OLD, REMAP_NEW),
    ):
        count = text.count(old)
        if count != 1:
            print(f"refusing: {label} match count={count}")
            return 2
        text = text.replace(old, new)

    TARGET.write_text(text)
    print(f"patched: {TARGET}")
    print(f"backup:  {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
