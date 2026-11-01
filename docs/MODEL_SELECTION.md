# 模型选型说明

## 1. 选了什么

| 项 | 值 |
|---|---|
| 模型 | `Qwen/Qwen3.8-27B-FP8`（赛题固定 Qwen3.8-27B，我们选官方 FP8 权重版本） |
| 版本 | Qwen3.8-27B-FP8（官方 FP8 发布版），ModelScope commit `416b6c4c` |
| 参数规模 | 27B，混合架构：注意力层 + Gated DeltaNet（GDN/Mamba 类）线性层 |
| 词表 / hidden | 248320 / 5120（由 lm_head 形状 `(248320, 5120)` 确定） |
| 权重大小 | 29 GB（FP8） |
| 附带模块 | `mtp.safetensors`（多 token 预测头）、视觉塔（本任务不用，`--language-model-only` 跳过） |
| 来源 | ModelScope `Qwen/Qwen3.8-27B-FP8`；平台共享模型库路径 `/model/ModelScope/Qwen/Qwen3.8-27B-FP8` |
| 量化格式 | **checkpoint 自带 FP8**：权重 FP8 E4M3，按 `[128, 128]` 块携带 FP32 scale，激活在线量化（W8A8），走 block-wise FP8 GEMM |
| 精度 | 其余计算 BF16；KV cache 用 `fp8_e4m3`；Mamba 状态 `bfloat16` |
| 上下文 | 服务侧 65536（官方 harness 要求 >32768） |
| 草稿模型 | `incoai/Qwen3.8-27B-DFlash2`（3.85 GB，commit `5cfe71fd`），投机算法 `DFLASH`，draft 13，权重在线转 FP8 |

lm_head 在官方 checkpoint 里仍是 BF16（2.5 GB）。我们用 `patches/apply_sglang_fp8_lm_head.py` 在加载时把它按 `[128, 128]` 块在线量化成 FP8——这是提交配置里唯一改变主模型权重精度的量化（KV 的 `fp8_e4m3` 是缓存 dtype，不改模型权重），因此单独跑过 IFBench（严格分不变，见 Benchmark 报告 §3）。

## 2. 为什么是这个组合

模型由赛题固定，选型只需决定权重版本：官方 FP8 版免二次量化，保留 MTP 头；29 GB 权重在 48 GB 卡上仍给 KV 与草稿回滚状态留下空间，因此优化重心在 decode 而不是装载。显存账（同一次提交配置的实测）：

| 项 | 占用 | 说明 |
|---|---:|---|
| FP8 权重 | 29 GB | 固定 |
| BF16 lm_head | 2.5 GB → FP8 后约 1.2 GB | 在线量化省出 1.27 GB，KV 从 70604 涨到 99863 token |
| 视觉塔 | 0.92 GB | `--language-model-only` 不加载；文本输入下 MRoPE 退化为 1D RoPE，等价 |
| DFlash2 草稿权重 | 3.85 GB | 换来 c1 +46%，值得 |
| 草稿回滚状态 | 约 2 GB（紧凑窗口） | ReplaySSM 代替每个 draft token 0.63 GB 的完整状态，KV 从 123585 涨到 283423 token |
| prefill 瞬时激活 | 约 2.2 GB 空闲 | `--mem-fraction-static 0.92` 已是上限，再高启动即 OOM |
| HTTP 父进程 CUDA 上下文 | 454 MiB | `scripts/launch_sglang_parent_off_gpu.py` 让父进程不建上下文 |
| Mamba 槽 | 40 槽 | 默认 15 槽会把并发静默限到 3 |

`--mem-fraction-static 0.92` 下 KV 与状态的分配合计约 15 GB，8 路满载 KV 需要 ≥69632 token，这与官方 8192×8 并发的工作负载刚好匹配。

## 3. 与任务匹配度

长文本：65536 上下文，实测 32000 与 60000 token 输入全部成功（Benchmark 报告 §4）。逻辑推理与代码生成：启用 `--reasoning-parser qwen3` 与 `--tool-call-parser qwen3_coder`；官方质量口径 IFBench `prompt_level_strict` 0.81（门槛 0.76），逐题输出在 `logs/artifacts/`。

## 4. 未采用的方案与理由

| 方案 | 为什么不用 |
|---|---|
| INT4 / NVFP4 量化 | 48 GB 下不缺显存；sm_89 没有 FP4 硬件支持，省下的显存换不到吞吐 |
| 自实现更低价 lm_head（4/3-bit） | 无质量证据；lm_head 直接决定输出分布，风险大于收益 |
| `Qwen3.5-27B-FP8` | 无 `mtp.safetensors`，失去内置投机头；仅作引擎兼容性备选 |
| Coder390、Swift-1.5 等后训练变体 | 不是赛题固定的同一模型 |
| 14B / 9B 换吞吐 | 违反「模型固定 qwen3.8-27B」的硬性要求 |
| vLLM 0.25.1 作为主线 | 同卡实测落后：官方工具 MTP3、每档 20 请求时 SGLang c1/2/4/8 高 12% / 9% / 10% / 26%（Benchmark 报告 §2 引擎对比）。vLLM 上投机解码还需 `FULL_DECODE_ONLY` 才能保住 CUDA 图 |

## 5. 框架取舍的代价

选 SGLang 0.5.22 换来 DFLASH/DSPARK 投机、可调 FP8 block GEMM tile、radix 前缀缓存与 Mamba 池控制，代价是 **sm_89 不在 SGLang 官方验证矩阵**（官方验证的是 H200 / RTX PRO 6000 / RTX 5090 / DGX Spark）。两个具体后果：镜像里的 flashinfer-cubin 与 flashinfer 0.7.1 版本不匹配，需要 `FLASHINFER_DISABLE_VERSION_CHECK=1`；FP8 GEMM 的 `SWAP_AB`/`SPLIT_K` 被硬编码为 sm90 专用，需要 `patches/apply_sglang_sm89_gemm.py` 解除。
