# 48GB 4090 换卡重建与复现（2026-10-30）

## 机器

- Compshare pod `pod-A
- RTX 4090 48GB（49140 MiB），driver 610.57.04，CUDA UMD 13.3
- EPYC 7A23（cgroup 20 核），90GB RAM；overlay 50GB（根盘重建即丢）
- 模型库 `/model` 为 UPFS 50TB 持久盘，含 `Qwen3.8-27B-FP8`

## 模型

`/model/ModelScope/Qwen/Qwen3.8-27B-FP8`（29GB，block 128×128 FP8，含 `mtp.safetensors`）。
`Qwen3.5-27B-FP8` 没有 MTP 权重，只作兼容备选。

## 环境（根盘丢失后需重装）

```bash
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-venv python3-pip git
python3 -m venv /root/infer/venv

# uv 静态二进制（本地项目内无 uv 时从 astral 安装）
# 本机的 uv 0.11.16 也可直接拷贝到 /root/uv

mkdir -p /root/infer/{wheels,logs,tmp,uv-cache,cache}
cd /root/infer/wheels
curl -L -O 'https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl'
curl -L -O 'https://download.pytorch.org/whl/cu128/torchvision-0.25.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl'
curl -L -O 'https://download.pytorch.org/whl/cu128/torchaudio-2.10.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl'

# 阿里云直连 18MB/s；代理只有 1–2MB/s，务必 unset proxy
env -u https_proxy -u http_proxy -u HTTPS_PROXY -u HTTP_PROXY \
  UV_CACHE_DIR=/root/infer/uv-cache TMPDIR=/root/infer/tmp \
  /root/uv pip install --python /root/infer/venv/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  /root/infer/wheels/torch-2.10.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
  /root/infer/wheels/torchvision-0.25.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
  /root/infer/wheels/torchaudio-2.10.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl

# vllm 0.17.1 wheel 可从阿里云 PyPI 镜像下载；本地项目 wheels/ 也有同版本
curl -L -o vllm-0.17.1-cp38-abi3-manylinux_2_31_x86_64.whl \
  'https://mirrors.aliyun.com/pypi/packages/18/28/f85e67b390082481298e27a5c9f1da540d2d5abb1a06a594545cdc320818/vllm-0.17.1-cp38-abi3-manylinux_2_31_x86_64.whl'
env -u https_proxy -u http_proxy -u HTTPS_PROXY -u HTTP_PROXY \
  UV_CACHE_DIR=/root/infer/uv-cache TMPDIR=/root/infer/tmp \
  /root/uv pip install --python /root/infer/venv/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  /root/infer/wheels/vllm-0.17.1-cp38-abi3-manylinux_2_31_x86_64.whl
```

FlashInfer JIT 需要 `ninja` 与 nvcc 在 PATH；启动脚本已加入：

```bash
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
```

## 最终候选启动

```bash
cd /root/infer
NUM_SPEC=3 MAX_NUM_SEQS=8 MAX_MODEL_LEN=65536 GPU_UTIL=0.90 \
MAX_BATCHED_TOKENS=16384 ATTN_BACKEND=TRITON_ATTN \
nohup ./scripts/run_vllm_4090_fp8.sh > logs/vllm_server.log 2>&1 &
```

关键配置：FP8 W8A8 block、MTP draft=3、`kv-cache-dtype=fp8`、`TRITON_ATTN`、
`compilation mode=0 + cudagraph FULL_DECODE_ONLY`（capture 1/2/4/8/16）、
`language-model-only`、`max-model-len=65536`、`max-num-seqs=8`、`max-num-batched-tokens=16384`。
KV 11.64GiB，65K 并发 4.23x，空载显存约 43GB。

不要用 `VLLM_TEST_FORCE_FP8_MARLIN=1` 作为默认：decode TPOT 22.6ms，但 prefill TTFT
从 2.1s 涨到 3.5s、c8 从 43.5 掉到 32.0 tok/s。

## 结果（`vllm bench` random 8192 in / 512 out，16 prompts）

| 配置 | c1 | c2 | c4 | c8 | c1 TPOT p50 |
|---|---:|---:|---:|---:|---:|
| 官方文档 no-MTP（seqs=4） | 26.22 | 42.96 | 65.78 | 65.79 | 31.4ms |
| MTP1（seqs=8, chunk8192） | 29.68 | 52.88 | 86.43 | 116.76 | 28.8ms |
| **MTP3（seqs=8, chunk16384）** | **43.87** | — | **109.06** | **112.30** | **16.6ms** |

MTP3 相对官方 no-MTP：c1 +67~72%，c4 +63~66%，c8 +69~71%；acceptance 74.9%。

## 质量（官方 EvalScope IFBench，limit=100，max_tokens=32768）

- `prompt_level_strict = 0.78`（78/100）
- `inst_level_strict = 0.7928`
- `prompt_level_loose = 0.85`
- 官方文档 FP8 no-MTP = 0.80；相对 97.5%，通过 ≥95% 门槛。
- gate 使用 batch=8；官方 harness 的 batch=2 复测尚未跑（换卡前只完成 gate）。
- 没有 IFBench 请求因 32768 截断；48 个 `length` 是性能测试 `ignore_eos` 造成的。

## 尚未完成（新卡优先）

1. 官方 harness AIPerf 100 请求 × c1/2/4/8（本地环境已备好）。
2. 官方 harness IFBench batch=2 复测。
3. 30–60min soak、32K/64K 长上下文稳定性。
4. 可选：FP8 block GEMM kernel config 调优（L40S JSON 可参考）。

## 本地 artifacts

`logs/artifacts/20261030_48g/`

- `eval/out_full_mtp3_b8/`：IFBench 100 题 predictions/reviews/reports
- `eval/ifbench_full_mtp3_b8.log`：完整运行日志
- `infer/logs/`：各配置 vLLM server 日志、`pip_freeze_*.txt`、`machine_env.txt`
- `infer/scripts/run_vllm_4090_fp8.sh`：启动脚本
- `dry.json` / `check.json`：官方 harness dry-run 与依赖检查