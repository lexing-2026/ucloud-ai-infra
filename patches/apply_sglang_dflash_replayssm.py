#!/usr/bin/env python3
"""GDN ReplaySSM spec-verify under DFlash (SGLang).

Target verify kept one full GDN state per draft token so a rejected draft could roll back: 0.63 GB
per draft token at 8 running requests, and at c8 about 6 GB of state writes per step, 10% of a step.
--enable-linear-replayssm-spec replaces those snapshots with a compact (d, k, g) window that the
accepted prefix folds into the checkpoint. KV capacity goes from 123585 to 283423 tokens.

SGLang wires that up for MTP only. DFlash called the backend commit directly, which knows the KDA
ring alone, so the configurator refused GDN models. The GDN commit is the generic
commit_mamba_states_after_verify and DFlash already has the step indices it takes.

Apply with the python of the SGLang env:
  /root/infer/sg22/bin/python patches/apply_sglang_dflash_replayssm.py
"""
import importlib.util
from pathlib import Path
import sys

EDITS = {
    "srt/speculative/dflash_worker_v2.py": (
        "        model_runner = self.target_worker.model_runner\n"
        '        if hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):\n'
        "            attn_backend.update_mamba_state_after_mtp_verify(\n"
        "                last_correct_step_indices=last_correct_step_indices,\n",
        "        model_runner = self.target_worker.model_runner\n"
        '        mamba_pool = getattr(model_runner.req_to_token_pool, "mamba_pool", None)\n'
        '        if getattr(mamba_pool, "replayssm_cache_base", None) is not None and not getattr(\n'
        '            mamba_pool, "replayssm_is_kda", False\n'
        "        ):\n"
        "            from sglang.srt.speculative.spec_utils import (\n"
        "                commit_mamba_states_after_verify,\n"
        "            )\n"
        "\n"
        "            commit_mamba_states_after_verify(\n"
        "                self.target_worker,\n"
        "                batch,\n"
        "                commit_lens,\n"
        "                commit_lens,\n"
        "                int(self.block_size),\n"
        "                prepared_step_indices=(last_correct_step_indices, mamba_steps_to_track),\n"
        "            )\n"
        "            return\n"
        '        if hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):\n'
        "            attn_backend.update_mamba_state_after_mtp_verify(\n"
        "                last_correct_step_indices=last_correct_step_indices,\n",
    ),
    "srt/mem_cache/kv_cache_configurator.py": (
        '            and _algo in ("DSPARK", "DFLASH")\n'
        "            and self.hybrid_kda_config is None\n",
        '            and _algo in ("DSPARK",)\n'
        "            and self.hybrid_kda_config is None\n",
    ),
}


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        print("sglang not found in this python", file=sys.stderr)
        return 1
    root = Path(spec.origin).parent
    for name, (old, new) in EDITS.items():
        path = root / name
        text = path.read_text()
        if new in text:
            print(f"already patched: {path}")
            continue
        if text.count(old) != 1:
            print(f"expected one match in {path}", file=sys.stderr)
            return 1
        path.write_text(text.replace(old, new))
        print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
