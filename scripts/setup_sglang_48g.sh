#!/usr/bin/env bash
set -euo pipefail

# Rebuild the serving env and official harness on a fresh Compshare 4090 48G pod
# (image: Ubuntu 22.04, CUDA 13.2, /usr/local/miniconda3/envs/py312).
# Mirrors are used because the default proxy only gives 1-2 MB/s.

BASE=${BASE:-/usr/local/miniconda3/envs/py312}
VENV=${VENV:-/root/infer/sg22}
HARNESS=${HARNESS:-$HOME/.cache/ai-infra-eval/envs}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/infer/uv-cache}

"$BASE/bin/python" -m pip install -q -i "$UV_DEFAULT_INDEX" uv
ln -sf "$BASE/bin/uv" /usr/local/bin/uv
ln -sf "$BASE/bin/python3.12" /usr/local/bin/python3.12

uv venv --no-project --python "$BASE/bin/python" --system-site-packages "$VENV"
uv pip install --python "$VENV/bin/python" 'sglang==0.5.22'
"$VENV/bin/python" -c "import sglang, torch; print(sglang.__version__, torch.__version__, torch.cuda.is_available())"

# RTX 4090 FP8 GEMM tiles, the sm89 gate removal they rely on, the FP8 lm_head, the hand kernels,
# the copy-free GDN prefill and the compact GDN verify state under DFlash.
REPO=$(cd "$(dirname "$0")/.." && pwd)
"$VENV/bin/python" "$REPO/patches/apply_rtx4090_fp8_gemm_configs.py"
"$VENV/bin/python" "$REPO/patches/apply_sglang_sm89_gemm.py"
"$VENV/bin/python" "$REPO/patches/apply_sglang_fp8_lm_head.py"
"$VENV/bin/python" "$REPO/patches/apply_sglang_fp8_hand_gemm.py"
"$VENV/bin/python" "$REPO/patches/apply_sglang_language_only.py"
"$VENV/bin/python" "$REPO/patches/apply_sglang_gdn_prefill.py"
"$VENV/bin/python" "$REPO/patches/apply_sglang_dflash_replayssm.py"

mkdir -p "$HARNESS"
uv venv --no-project --python "$BASE/bin/python" "$HARNESS/aiperf-0.13.0"
uv venv --no-project --python "$BASE/bin/python" "$HARNESS/evalscope-1.13.0"
uv pip install --python "$HARNESS/aiperf-0.13.0/bin/python" 'aiperf==0.13.0'
uv pip install --python "$HARNESS/evalscope-1.13.0/bin/python" 'evalscope[ifbench]==1.13.0' emoji syllapy nltk
"$HARNESS/evalscope-1.13.0/bin/python" -m nltk.downloader -d "$HARNESS/nltk_data" \
  punkt_tab stopwords averaged_perceptron_tagger_eng

# DSpark draft weights are not on the shared /model volume (DFlash2 is).
# Only needed for SPEC=dspark; skip with DSPARK_DOWNLOAD=0 (curl -C - exits non-zero
# on an already-complete file, which would abort the rest of a re-run).
DSPARK=${DSPARK:-/root/infer/models/Qwen3.8-27B-DSpark}
if [ "${DSPARK_DOWNLOAD:-1}" = 1 ]; then
  DSPARK_URL='https://modelscope.cn/api/v1/models/RadixArk/Qwen3.8-27B-DSpark/repo?Revision=master&FilePath='
  mkdir -p "$DSPARK"
  for f in README.md config.json dflash.py dspark.py configuration.json model.safetensors; do
    curl -sSL -C - --retry 5 -o "$DSPARK/$f" "$DSPARK_URL$f"
  done
  echo "2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7  $DSPARK/model.safetensors" | sha256sum -c -
fi
