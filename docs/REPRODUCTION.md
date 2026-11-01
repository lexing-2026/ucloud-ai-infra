# 完整复验材料包

## 0. 版本快照

- 代码快照：本仓库即快照，Git 标签与提交 hash 见 `VERSION`；GitHub 远端与 tag 同名。
- 模型、引擎、依赖清单与逐文件 sha256：见仓库根 `VERSION`。
- 依赖全量：`logs/artifacts/20261030_48g/official/pip_freeze_sg22.txt`。

## 1. 模型文件与获取方式

| 用途 | 来源 | 大小 | 校验 |
|---|---|---:|---|
| 主模型 | ModelScope `Qwen/Qwen3.8-27B-FP8`，commit `416b6c4c`；平台共享库 `/model/ModelScope/Qwen/Qwen3.8-27B-FP8` | 29 GB | 加载日志：`type=Qwen3_5ForConditionalGeneration, quant=fp8, fmt=e4m3`，权重实际占 27.24 GB |
| 草稿模型（提交配置） | ModelScope `incoai/Qwen3.8-27B-DFlash2`，commit `5cfe71fd`；平台路径 `/model/ModelScope/incoai/Qwen3.8-27B-DFlash2` | 3.85 GB（加载实测 3.69 GB） | 平台 `/model` 已含，无需下载 |
| 对照草稿 | ModelScope `RadixArk/Qwen3.8-27B-DSpark`，不在共享库内 | — | `scripts/setup_sglang_48g.sh` 下载 6 个文件并校验 `model.safetensors` 的 sha256 = `2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7`；提交配置用不到它，`DSPARK_DOWNLOAD=0` 可跳过 |

若复验环境没有 `/model`，下面的脚本按固定提交下载两个仓库，并用 `scripts/models.sha256`（80 个文件）校验：

```bash
bash scripts/get_models.sh
```

然后把 `MODEL` 与 `DRAFT_DFLASH` 指到 `/root/infer/models` 下的两个目录（`scripts/run_sglang_48g.sh` 头部列出了全部变量）。

平台共享库可以用同一份清单核对：

```bash
mkdir -p /tmp/v
ln -sfn /model/ModelScope/Qwen/Qwen3.8-27B-FP8 /tmp/v/Qwen3.8-27B-FP8
ln -sfn /model/ModelScope/incoai/Qwen3.8-27B-DFlash2 /tmp/v/Qwen3.8-27B-DFlash2
(cd /tmp/v && sha256sum --quiet -c /root/infer/scripts/models.sha256)
```

提交号与 sha256 取自 ModelScope 文件接口。两个仓库的最新提交分别在 2026-08-14 和 2026-08-20，早于本项目全部实验。

## 2. 环境与硬件前置

| 项 | 要求 |
|---|---|
| GPU | 1 × RTX 4090 48GB（49140 MiB），sm_89；无需 NVLink、无需多卡 |
| 驱动 | ≥ 610.57.04（我们用的 driver + CUDA 13.2 UMD）；容器需暴露 `/dev/nvidia*` |
| 镜像 | 优云智算 `SGLang v0.5.15` 镜像（Ubuntu 22.04 / CUDA 13.2 / Python 3.12.11 / torch 2.11.0+cu130 / ptxas）；其他 Ubuntu 22.04 + CUDA 13.x 镜像可用，但需自行准备 torch ≥2.11 的 cu130 轮子 |
| 磁盘 | ≥ 60 GB（venv 约 3 GB + uv 缓存 + 草稿权重） |
| 网络 | 能访问 PyPI 镜像与 ModelScope；`setup_sglang_48g.sh` 默认走 `mirrors.aliyun.com` |
| 编译工具 | `ptxas`（打手写 kernel 补丁时用，`-O1` 预编译 cubin） |

## 3. 从零到提交配置（三条命令）

```bash
git clone https://github.com/lexing-2026/ucloud-ai-infra /root/infer && cd /root/infer        # 路径不是 /root/infer 时用 VENV/LOGDIR 覆盖
bash scripts/setup_sglang_48g.sh                        # venv + 引擎 + 7 个补丁 + 评测工具 + DSpark
bash scripts/run_sglang_48g.sh                          # 不带参数即提交配置，端口 30000，约 3 分钟
```

冒烟与自检：

```bash
curl -s http://127.0.0.1:30000/v1/models
bash scripts/demo_live.sh
bash docs/eval/run.sh --mode check          # 只检查评测工具依赖，不请求模型
```

补丁安装顺序及作用（`setup_sglang_48g.sh` 已按此顺序执行，单独重跑亦可）：

```bash
V=/root/infer/sg22/bin/python
$V patches/apply_rtx4090_fp8_gemm_configs.py   # 安装调好的 4090 FP8 GEMM tile
$V patches/apply_sglang_sm89_gemm.py           # 解除 SWAP_AB / SPLIT_K 的 sm90 限制
$V patches/apply_sglang_fp8_lm_head.py         # 在线 FP8 lm_head（开关 SGLANG_FP8_LM_HEAD=1）
$V patches/apply_sglang_fp8_hand_gemm.py       # 手写 sm_89 GEMM 与融合 MLP，ptxas -O1 预编译
$V patches/apply_sglang_language_only.py       # 允许 Qwen3.5/3.8 走 --language-model-only
$V patches/apply_sglang_gdn_prefill.py         # GDN prefill 去拷贝并融合 conv、归一化、o、norm 量化
$V patches/apply_sglang_dflash_replayssm.py    # 让 DFlash 走 GDN ReplaySSM 验证
```

每个补丁都是幂等的：重复执行会检测已打并跳过。手写 kernel 有回退保护——任何不满足条件的调用（融合 kernel 的 M 低于 1024、形状不在覆盖表、非 sm_89）都原样走 Triton，因此关掉 `SGLANG_FP8_HAND_GEMM` 即回到纯 Triton 路径，可用来做 A/B。

## 4. 复测命令与预期数字

```bash
bash docs/eval/run.sh --base-url http://127.0.0.1:30000/v1 \
  --model qwen3.8-27b-fp8 --mode all
```

| 指标 | 提交结果（C 机，100 请求/档） | 复验容差 | 耗时 |
|---|---|---|---:|
| c1 / c2 / c4 / c8 输出吞吐 | 138.99 / 190.98 / 234.42 / 254.39 tok/s | ±3%（同机同镜像）；跨卡 ±4~6% | 约 20 分钟 |
| c1 TTFT P50 | 1.569 s | 同型号卡之间差约 4% | — |
| c1 TPOT P50 / c8 TPOT P50 | 3.67 ms / 20.55 ms | ±10%（受功耗墙与排队影响） | — |
| 错误请求 | 0 | 必须为 0 | — |
| IFBench `prompt_level_strict` | 0.83（门槛 0.76） | ±0.04：未设 temperature，100 题采样本身会抖 | 约 43 分钟 |
| 显存峰值 / KV 容量 | 47338 MiB / `max_total_num_tokens=283423` | 峰值随 `--mem-fraction-static` 固定；KV 受卡差影响 | — |

提交配置的 KV 与显存证据在 `logs/artifacts/20261031_final/C/logs/z13_apps.txt`。

1. **同机同口径**才有可比性。官方 AIPerf（100 请求 + 8 预热）与我们自己的 `scripts/bench_ab.py`（随机 token-ID）数字不可互相换算；20 请求与 100 请求也不行（FP8 lm_head 的 c1 在 20 请求时是 112.78，100 请求时是 106.24）。
2. 吞吐明显低于容差时，先查四件事：`nvidia-smi` 是否有其他进程、`--max-mamba-cache-size` 是否被默认值吃掉（并发会掉到 3）、是否走了 Triton 而非手写路径（看启动日志有没有 hand kernel 装载行）、`--mem-fraction-static` 是否被改动。
3. 手写路径在大 M 才生效（GEMM M≥6144、融合 kernel M≥1024）。IFBench 的短 prompt 走不到，所以强制口径 `SGLANG_FP8_HAND_MIN_M=100` 才是测到手写路径的质量门禁；默认阈值下那次的分数只能说明 Triton 路径。

## 5. 逐条证据怎么找

- 报告里每张表 → `docs/BENCHMARK_REPORT.md` 第 6 节 → `docs/EVIDENCE_INDEX.md`（机器生成，含每个运行的请求数、并发、吞吐、TTFT/TPOT、错误数、质量分）→ 目录内的 `summary.json` / `report.md` / `performance/c*/profile_export.jsonl`（逐请求）。
- 被否决的尝试与原因 → [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) §6 失败案例表（15 条）。
- 手写 kernel 的数值等价性与搜索过程 → `logs/artifacts/20261031_hand/C/logs/{eqv_hand.json,eqv_tri.json,eqv_cmp.txt,search*.txt,fsearch1.txt,hand3.txt,minm.txt}` 与同目录上层的 `*.py`。
- tile 调优全过程 → `logs/artifacts/20261030_48g/official/fp8_gemm_tuning_log*.ndjson`、`20261031_hand/fp8_gemm_tuning_log_d10*.ndjson`，复现命令 `python3 scripts/tune_fp8_gemm.py --help`。

## 6. 复验环境差异说明

| 项 | 说明 |
|---|---|
| 卡间差异 | 同型号两张 4090 48GB 的 c1 TTFT 恒差约 4%（1.90 s 对 1.82 s），因此对比一律同机 |
| 功耗墙 | 持续满载顶 450 W，纯 TC 峰值只要 259 W，多出的约 190 W 在搬数据；短时单次计时会被升频骗到，所有算子级取舍都用 2 秒以上持续满载。复验环境若功耗上限不同，算子级百分比会漂移 |
| 时钟不可调 | 容器内 `nvidia-smi -lgc/-ac` 返回成功但不生效，我们未依赖任何改时钟的操作 |
| sm_89 未官方验证 | SGLang 0.5.22 的验证矩阵是 H200/RTX PRO 6000/RTX 5090/DGX Spark；本方案需要 `FLASHINFER_DISABLE_VERSION_CHECK=1` 与两个补丁解除 sm90 限制，属于自行承担风险的配置 |
| 已删除的补丁 | GDN prefill 分段（省 0.7 GB 但 c1 TTFT +4%）与 GEMM K 循环精确改写（收益低于噪声）的补丁已从仓库删除，结论见优化报告 §6 |
