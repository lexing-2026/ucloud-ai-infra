#!/usr/bin/env bash
set -euo pipefail

# Primary serving config: Qwen3.8-27B-FP8 on 1x RTX 4090 48G with SGLang 0.5.22.
# Build the env with scripts/setup_sglang_48g.sh, then run this with no arguments.
# sm89 is NOT in SGLang's validated matrix (H200/RTX PRO 6000/RTX 5090/DGX Spark).
#
# Env knobs:
#   SPEC=off|mtp|dflash|dspark
#   CONTEXT=65536          must stay >32768 for the official harness
#   MEM_FRAC=0.92          --mem-fraction-static
#   KV=fp8_e4m3
#   ATTN=flashinfer|triton
#   MAMBA_DTYPE=bfloat16|float32
#   MAMBA_RATIO=...        --mamba-full-memory-ratio (see calculator)
#   MAX_MAMBA=...          --max-mamba-cache-size (overrides ratio)
#   CHUNK=8192             --chunked-prefill-size
#   MAX_RUNNING=8          --max-running-requests
#   DRAFT_WINDOW=2048      --speculative-draft-window-size for DFlash2, 0 = none
#   DRAFT_QUANT=fp8        --speculative-draft-model-quantization for DFlash2, empty = BF16 draft
#   REPLAY_SPEC=1          --enable-linear-replayssm-spec for DFlash2, needs patches/apply_sglang_dflash_replayssm.py
#   VISION=0               1 keeps the vision tower (0.92 GB) for image requests. Images are then
#                          preprocessed on CPU: the HTTP parent has no CUDA context.
# Not a knob: --enable-cache-report is permanent. Without it the OpenAI usage payload has
# prompt_tokens_details=null, so agent clients reading cached_tokens (dsh) show a 0% cache hit
# even while the radix cache hits ~99%.
#   SGLANG_FP8_LM_HEAD=1   online FP8 lm_head, needs patches/apply_sglang_fp8_lm_head.py
#   SGLANG_FP8_HAND_GEMM=1 tile first FP8 weights, hand-scheduled sm_89 GEMM and fused MLP kernel,
#                          needs patches/apply_sglang_fp8_hand_gemm.py
#
# Examples:
#   bash scripts/run_sglang_48g.sh              # DFlash2 draft 13 + FP8 draft + tile + FP8 lm_head + hand kernels + ReplaySSM verify, the submitted config
#   SPEC=mtp bash scripts/run_sglang_48g.sh
#   SPEC=off bash scripts/run_sglang_48g.sh

VENV=${VENV:-/root/infer/sg22}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen3.8-27B-FP8}
DRAFT_DFLASH=${DRAFT_DFLASH:-/model/ModelScope/incoai/Qwen3.8-27B-DFlash2}
DRAFT_DSPARK=${DRAFT_DSPARK:-/root/infer/models/Qwen3.8-27B-DSpark}   # RadixArk/Qwen3.8-27B-DSpark on ModelScope
PORT=${PORT:-30000}
SERVED=${SERVED:-qwen3.8-27b-fp8}
CONTEXT=${CONTEXT:-65536}
MEM_FRAC=${MEM_FRAC:-0.92}
KV=${KV:-fp8_e4m3}
ATTN=${ATTN:-flashinfer}
MAMBA_DTYPE=${MAMBA_DTYPE:-bfloat16}
MAMBA_RATIO=${MAMBA_RATIO:-}
# The Mamba pool caps concurrency silently: 15 slots means 3 running requests.
MAX_MAMBA=${MAX_MAMBA:-40}
MAX_RUNNING=${MAX_RUNNING:-8}
CHUNK=${CHUNK:-8192}
SPEC=${SPEC:-dflash}
# Draft 13 won the 100 request runs on the stock FP8 draft (x10/x12/x13 in logs/iterations_b.md).
DRAFT_TOKENS=${DRAFT_TOKENS:-13}
# No measurable gain over 0 at 100 requests; kept because IFBench and the soak ran with it.
DRAFT_WINDOW=${DRAFT_WINDOW:-2048}
DRAFT_QUANT=${DRAFT_QUANT-fp8}
REPLAY_SPEC=${REPLAY_SPEC:-1}
DRAFT_KV=${DRAFT_KV:-$KV}
MTP_STEPS=${MTP_STEPS:-4}
# off = --cuda-graph-backend-prefill disabled; the DFlash2/DSpark runs used off.
PREFILL_GRAPH=${PREFILL_GRAPH:-off}
VISION=${VISION:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/root/infer/cache}
export HF_HOME=${HF_HOME:-/root/infer/hf}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# The image's flashinfer-cubin (0.6.x) has no 0.7.1 release; Triton attention is used.
export FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-1}
export SGLANG_FP8_LM_HEAD=${SGLANG_FP8_LM_HEAD:-1}
export SGLANG_FP8_HAND_GEMM=${SGLANG_FP8_HAND_GEMM:-1}
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"

SPEC_ARGS=()
case "$SPEC" in
  off) ;;
  mtp)    SPEC_ARGS=(--speculative-algorithm EAGLE
                     --speculative-num-steps "$MTP_STEPS"
                     --speculative-eagle-topk 1
                     --speculative-num-draft-tokens "$((MTP_STEPS + 1))") ;;
  dflash) SPEC_ARGS=(--speculative-algorithm DFLASH
                     --speculative-draft-model-path "$DRAFT_DFLASH"
                     --speculative-num-draft-tokens "$DRAFT_TOKENS"
                     --speculative-draft-kv-cache-dtype "$DRAFT_KV")
          [ "$DRAFT_WINDOW" = 0 ] || SPEC_ARGS+=(--speculative-draft-window-size "$DRAFT_WINDOW")
          [ -z "$DRAFT_QUANT" ] || SPEC_ARGS+=(--speculative-draft-model-quantization "$DRAFT_QUANT")
          [ "$REPLAY_SPEC" = 0 ] || SPEC_ARGS+=(--enable-linear-replayssm-spec) ;;
  dspark) SPEC_ARGS=(--speculative-algorithm DSPARK
                     --speculative-draft-model-path "$DRAFT_DSPARK"
                     --speculative-draft-model-quantization unquant
                     --speculative-draft-attention-backend "$ATTN"
                     --speculative-dspark-block-size "$DRAFT_TOKENS"
                     --speculative-num-steps 1
                     --speculative-eagle-topk 1
                     --speculative-draft-kv-cache-dtype "$KV") ;;
  *) echo "unknown SPEC=$SPEC" >&2; exit 2 ;;
esac

# extra_buffer_lazy asserts under speculation; no_buffer disables overlap scheduling.
MAMBA_STRATEGY=${MAMBA_STRATEGY:-extra_buffer}
MAMBA_ARGS=(--mamba-ssm-dtype "$MAMBA_DTYPE" --mamba-radix-cache-strategy "$MAMBA_STRATEGY")
[ -n "$MAMBA_RATIO" ] && MAMBA_ARGS+=(--mamba-full-memory-ratio "$MAMBA_RATIO")
[ -n "$MAX_MAMBA" ] && MAMBA_ARGS+=(--max-mamba-cache-size "$MAX_MAMBA")
LANG_ARGS=(--language-model-only)
[ "$VISION" = 1 ] && LANG_ARGS=(--image-processor-backend pil)
GRAPH_ARGS=()
[ "$PREFILL_GRAPH" = off ] && GRAPH_ARGS=(--cuda-graph-backend-prefill disabled)

echo "sglang: ctx=$CONTEXT mem=$MEM_FRAC kv=$KV attn=$ATTN mamba=$MAMBA_DTYPE ratio=${MAMBA_RATIO:-auto} chunk=$CHUNK spec=$SPEC prefill_graph=$PREFILL_GRAPH"

# The wrapper keeps the HTTP parent off the GPU, 454 MiB back for the scheduler.
exec "$VENV/bin/python" "$(dirname "$(readlink -f "$0")")/launch_sglang_parent_off_gpu.py" \
  --model-path "$MODEL" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --tp-size 1 \
  --context-length "$CONTEXT" \
  --mem-fraction-static "$MEM_FRAC" \
  --kv-cache-dtype "$KV" \
  --attention-backend "$ATTN" \
  --chunked-prefill-size "$CHUNK" \
  --max-running-requests "$MAX_RUNNING" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-cache-report \
  "${LANG_ARGS[@]}" \
  "${MAMBA_ARGS[@]}" \
  ${GRAPH_ARGS[@]+"${GRAPH_ARGS[@]}"} \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  ${EXTRA_ARGS}
