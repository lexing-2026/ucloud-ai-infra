# 优化方案与性能分析

硬件：1 × RTX 4090 48GB（sm89）。模型：`Qwen/Qwen3.8-27B-FP8`。数据与测试条件见 [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)。

**结论**：官方 AIPerf 每档 100 请求，c1/2/4/8 = 138.99 / 190.98 / 234.42 / 254.39 tok/s，零错误，IFBench 0.83（门槛 0.76）。相对原样 vLLM 提升 7.6× / 5.2× / 3.6× / 3.9×。提升来自五层：换引擎、sm89 tile、DFlash2 投机、手写算子。第 3、4 节是逐步优化的过程，其中的 draft 8、mem 0.925 是当时的配置；最终配置以第 5 节为准。

## 1. 模型与引擎选型

- **模型**：赛题固定 Qwen3.8-27B。选官方 FP8 版（29GB）：48GB 显存放得下权重、KV 和 Mamba 状态，并带 `mtp.safetensors`，可直接做投机解码。
- **未采用的模型**：Coder390、Swift-1.5 等后训练变体不是同一模型；INT4/NVFP4 省显存，但 48GB 下不缺显存，且 sm89 没有 FP4 硬件支持。
- **引擎**：vLLM 与 SGLang 同卡实测后选 SGLang 0.5.22。官方工具每档 20 请求、MTP3 时，SGLang c1/2/4/8 比 vLLM 0.25.1 高 12% / 9% / 10% / 26%。

## 2. 瓶颈分析

一次 decode 要读完全部权重，FP8 权重约 29GB，4090 显存带宽约 1008GB/s，理论下限约 29ms/步。投机解码能减少步数，但每步要验证更多 token，所以瓶颈分三处：

1. **FP8 GEMM 既是 prefill 的主成本，又在小 M 下浪费算力。** 8K prefill 的逐 kernel profile：FP8 GEMM 1305ms，占 prefill 的 73.8%；attention 6.7%，GDN 约 6%。4090 上两个引擎都落到 Triton block GEMM，且都没有 4090 配置，全部走默认 `BLOCK_M=64`。kernel 用 `% M` 把行数补满 tile，MTP4 单请求只有 M=5，却要按 64 行计算。
2. **并发被静默限流。** SGLang 默认 Mamba 池只有 15 槽，每请求占 5 槽，日志一行 warning 就把并发限到 3，c8 的 TTFT 因此到 24s。
3. **prefill 占用 c8 大半时间。** 单请求 8192 token 约 1.8s，c8 时每请求约 2.4s，其中四分之三是 prefill。
4. **prefill 的激活张量走满 HBM。** 同一份 prefill 时间线里 GPU 忙碌占 99.4%，CPU 调度不是瓶颈。`act_and_mul`、激活量化、rmsnorm 的折合带宽是 780–970 GB/s：8192 行的激活放不进 L2，单层 MLP 仅激活就要在 HBM 走 1.86 GB。
5. **decode 里还有一块 BF16。** 官方权重只有 lm_head 仍是 BF16（2.5GB），DFlash 草稿借用主模型的 head，所以每步读两次，占一步 11–13%，单次开销与 5 层草稿相当。

## 3. 优化措施

| 措施 | 针对瓶颈 | 结果 |
|---|---|---|
| CUDA 全图（SGLang decode/verify） | Python 与 kernel launch 开销 | vLLM 上 c1 TPOT 从 eager 的 32.55ms 到 17.69ms |
| MTP 投机解码，深度 4 | 减少 decode 步数 | 见下表，2/3/5/7 均更差或失衡 |
| 放开 Mamba 池到 40 槽 | 并发限流 | c8 从 103.84 到 137.77 tok/s |
| FlashInfer attention | 验证与 prefill 的 attention | c8 从 137.63 到 150.21 tok/s |
| **FP8 GEMM tile 调优 + SWAP_AB/SPLIT_K** | 小 M 算力浪费 | c1 +30%，c8 +14%（同客户端） |
| 第二代 tile：补 M=9/72 锚点 | 投机验证批的 tile 选择 | c8 +3.6~6.7%（同机同口径） |
| DFlash2 draft 8 投机 | decode 步数 | 官方口径 c1 69.62 → 101.52（+46%），c8 +10% |
| DSpark block 7 投机 | decode 步数 | 官方口径 c1 93.56，c8 202.53；c8 与 DFlash2 打平 |
| **在线 FP8 lm_head** | decode 的 BF16 权重读取 | 同机官方 100 请求 c1 +1%、c2 +2.3%、c4 +1.8%、c8 +1.8%，IFBench 不变 |
| FP8 KV + 65536 上下文 | 显存与长上下文 | 60K 输入稳定运行 |
| Radix 前缀缓存 | 共享前缀场景的 prefill | 7168 token 公共前缀时 c1 TTFT 1.88s → 0.31s（token-ID 输入） |

投机深度扫描（`scripts/bench_ab.py`，8192/512，单轮小样本，c1 跨轮波动约 ±5%）：

| 深度 | 无 tile c1 | 无 tile c8 | 有 tile c1 | 有 tile c8 |
|---:|---:|---:|---:|---:|
| 2 | 41.13 | 119.88 | — | — |
| 3 | 46.15 | 137.77 | 59.85 | 166.73 |
| **4** | 48.75 | 137.63 | **61.98** | **171.20** |
| 5 | 47.74 | 122.34 | 66.37 | 158.40 |
| 7 | 37.89 | 48.91 | — | — |

深度 5 的 c1 更高，但 c8 退化，官方四档综合选深度 4。

### 草稿模型：DFlash2 与 DSpark

MTP 之外，SGLang 0.5.22 还支持 DFLASH/DSPARK。两者都用官方文本输入、每档 100 请求复测：

| 配置 | 草稿 | 显存比例 | c1 | c2 | c4 | c8 | 接受率 |
|---|---|---:|---:|---:|---:|---:|---:|
| MTP4 + tile | 内置 MTP | 0.90 | 69.62 | 105.91 | 148.12 | 184.41 | — |
| DSpark block 7 + tile | RadixArk DSpark | 0.92 | 93.56 | 138.37 | 177.38 | 202.53 | 0.726 |
| DFlash2 draft 8 + tile | incoai DFlash2 | 0.925 | 101.52 | 142.37 | 183.45 | 203.25 | 0.715 |
| DFlash2 draft 8 + 第二代 tile | incoai DFlash2 | 0.925 | 105.25 | 148.11 | 189.68 | 212.22 | — |
| + 在线 FP8 lm_head + 窗口 2048（上一版提交） | incoai DFlash2 | 0.925 | 106.24 | 152.11 | 192.38 | 215.88 | 0.815 |

同一 tile 下，DFlash2 draft 8 在 c1/c2/c4 最高，c8 与 DSpark 打平；补上 M=9/72 的第二代 tile 后四档都最高。代价是 c8 的 TTFT P50 从 1.97s 升到 4.47–4.74s：草稿权重与草稿 KV 多占显存，KV 容量降到约 7 万 token，100 请求下排队更早出现；但 c8 端到端从 21.4s 降到 19.7s。draft 12 只剩 4941 token KV，直接淘汰。

质量上 DFlash2 的 IFBench `prompt_level_strict` 为 0.79，DSpark 为 0.84，都高于 0.76 门槛（官方 FP8 无 MTP 为 0.80 的 95%）。按预设规则——过线即选最快——**提交配置取 DFlash2 draft 8**。

### decode 分解与在线 FP8 lm_head

逐 kernel profile（CUDA 图内 kernel 可见，每步均值）：

| 组成 | c1（约 41 ms/步） | c8（约 57 ms/步） |
|---|---:|---:|
| 主模型 FP8 GEMM | 27.3ms | 31.3ms |
| lm_head，每步两次 | 2×2.6ms | 2×3.1ms |
| 草稿 5 层 | 约 3.6ms | — |
| GDN 门控 | 0.6ms | 3.9ms |

`patches/apply_sglang_fp8_lm_head.py` 在加载时把 lm_head 按 [128, 128] 块量化成 FP8，走已调优的 W8A8 GEMM。SGLang 对量化 head 有原生分发，CUDA 图和 DFlash 的候选 top-k 都不受影响。单次 lm_head 从 2.63ms 降到 1.39ms，腾出 1.27GB 显存，KV 从 70604 涨到 99863 token。这一改动确实改变主模型 logits，所以单独过了 IFBench：严格分仍为 0.79，`inst_level_strict` 0.8108。

同机 100 请求对比（C 机）：

| 配置 | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|
| 基线 | 105.25 | 148.11 | 189.68 | 212.22 |
| FP8 lm_head | 106.31 | 151.56 | 193.14 | 216.02 |
| FP8 lm_head + 窗口 2048 | 106.24 | 152.11 | 192.38 | 215.88 |

A 机 FP8 lm_head 下有无窗口为 104.79 / 147.14 / 188.32 / 209.44 与 104.43 / 146.91 / 187.16 / 209.14。两台机顺序相反，差异都在 ±0.6% 内，窗口没有可测收益。保留它只因为 IFBench、soak 和 100 请求都在同一配置上完成，提交时不再换变量。

早期“DFlash2 已否决”的结论来自 vLLM 0.25.1 + 随机 token-ID 输入：接受率只有 0.20–0.39，且 draft 4 在 8 路下 OOM。官方文本输入下接受率是 0.715，结论反转。

## 4. 核心创新：sm89 FP8 GEMM 自动调优

方法借鉴 [ForgeKernel](https://github.com/OpenBMB/ForgeKernel)：以默认 tile 为 reference，候选先过正确性判定，再与 reference 在同一进程内计时，超过噪声阈值才晋升为新 reference，直到没有候选能再赢。ForgeKernel 的 kernel 面向 sm90/sm100 的 CuTe DSL，不能直接用于 4090，这里用的是它的流程。实现见 `scripts/tune_fp8_gemm.py`。

- **搜索空间**：`BLOCK_M/N/K`、`GROUP_M`、warps、stages，以及两个结构性开关。
- **结构性开关**：`SWAP_AB` 转置 MMA，使 M 维可降到 8；`SPLIT_K` 在 N/M 给出的并行块不足时沿 K 并行。SGLang 把两者限制在 sm90，但 kernel 本身不依赖 sm90，`patches/apply_sglang_sm89_gemm.py` 解除了这个限制。
- **正确性门禁**：与 reference 的平均相对误差须 ≤1e-3。判定器拦下过算错的 `BLOCK_K=256` 候选。
- **算子级结果**（同进程，M=5）：

| 形状 | 默认 | 调优后 | 提速 |
|---|---:|---:|---:|
| N=34816, K=5120 | 0.298ms | 0.248ms | 1.20× |
| N=16384, K=5120 | 0.151ms | 0.132ms | 1.15× |
| N=14336, K=5120 | 0.122ms | 0.076ms | 1.61× |
| N=5120, K=17408 | 0.214ms | 0.154ms | 1.39× |
| N=5120, K=6144 | 0.113ms | 0.068ms | 1.66× |

prefill（M=8192）五个形状提升 6–8%。

**unseen shape 检查发现过一个问题**：引擎按最近的 M 取配置，M≈4096 时离 40 比离 8192 近，prefill 误用了 decode tile，chunk 4096 时 TTFT 从 1.9s 涨到 3.6s。加 M=256 锚点后，128 到 16384 的输入 TTFT 都随长度线性增长。

**第二代：按投机验证批补 M=9/72**。投机解码每步验证 `c × (草稿数 + 1)` 个 token：单路 × 9 = 9，8 路 × 9 = 72。原网格只有 5/10/20/40/64，服务时 M=72 被映射到邻近档位，五个 decode 形状里有四个不是最优：

| 形状 | M=72 原选择 | 补锚点后 | 提速 |
|---|---|---:|---:|
| N=5120, K=6144 | 0.0833ms | 0.0533ms | 1.56× |
| N=5120, K=17408 | 0.1432ms | 0.1104ms | 1.30× |
| N=16384, K=5120 | 0.1154ms | 0.1012ms | 1.14× |
| N=34816, K=5120 | 0.2269ms | 0.2151ms | 1.06× |
| N=14336, K=5120 | 0.0785ms | 0.0776ms | 1.01× |

M=9 档原本已接近最优，补锚点只再快 0.1–6.5%。合计 decode GEMM 时间约降 14%。

服务级同机同口径 A/B（DFlash2 draft 8，`scripts/bench_ab.py`，8192/512）：c8 从 156.46 到 167.00 / 162.13 tok/s（两轮，+3.6~6.7%），TPOT p50 从 33.54ms 降到 31.00 / 30.77ms；c1 66.33 对 68.44（−3.1%，在噪声内）。decode GEMM 只占一步时间的一部分，所以服务端增益小于算子级。

**第三代：按 draft 10 的验证批补 M=11–88**。draft 从 8 改到 10 后验证批变成 c × 11，原来的 M=9/72 锚点全部失配。补锚点后 M=11 提速 1.08–1.44×，M=88 提速 1.02–1.29×，K≠5120 的形状普遍约 1.3×；lm_head 补 M=10–80，提速 1.02–1.05×。服务级吞吐变化在噪声内，c8 的 TPOT P50 从 24.78ms 降到 22.74ms。decode 受 HBM 带宽限制，算子级增益大部分到不了服务级。

## 5. 手写算子：sm_89 PTX GEMM 与融合 MLP

Triton 的 kernel 已经调过 tile，但这一层之下还有它表达不了的东西。手写版是 `patches/fp8_hand_sm89.py`，用 Python 生成 PTX，打补丁时用 `ptxas -O1` 预编译，请求路径只加载 `.cubin`。任何不满足条件的调用都原样回到 Triton。

### 决定成败的三条规则

这三条错了结果依然完全正确，只是慢。所以正确性测试发现不了它们。

| 版本 | 相对 Triton | 原因 |
|---|---:|---|
| 首版（线性 SMEM，每 lane 读不同行） | 0.40× | 一条 `cp.async` 的 32 个 lane 落在 32 条 cache line，L2 请求无法合并 |
| 每条 `cp.async` 覆盖 4 条 128B cache line | 0.65–0.70× | 请求合并后还剩 8 路 SMEM bank 冲突 |
| 加 `K_SW128`（`chunk ^= row & 7`） | 1.04–1.11× | `ldmatrix` 每相无冲突 |
| 随机搜索约 100 组后（gap、加载位置、ptxas 等级、tile、group、NB） | **1.07–1.09×** | `-O1` 优于 `-O3`，ptxas 的重排反而有害 |

### 比 Triton 多做了什么

- **软件流水**：Triton 的 kernel 跑完 64 条 mma 才开始 FP32 段。微基准测得 TC 忙时每条 mma 还能同时发射约 12 条整数指令，或 2 条 FMA（−1.2%）。手写版把缩放穿插进最后一个 K32 步。
- **FP 指令减半**：每 warp 每 tile 从 64 FMUL + 64 FFMA 减到 8 FMUL + 64 FFMA（`s[i]=a_s[i]*b_s` 只算一次）。
- **去掉第二次 barrier**：scale 不再走每 tile 的 SMEM 广播。
- **tile 取 128×128×128**：一个 B tile 由两倍的行共享，每 MAC 片上流量少 33%。

### 真正的约束是功耗，不是占用率

持续满载时 Triton 和手写版都顶在 450W。纯 TC 峰值只要 259W，多出来的约 190W 都在搬数据。所以占用率和时钟此消彼长：同样 450W，64 行 tile 约 2535 MHz，128 行 tile 约 2655 MHz。决定吞吐的是每 MAC 能耗。这也是早先“手写无法稳定超过 Triton”判断错的地方：只比了占用率上限，没有写出来比。短时单次计时会被升频骗到，所有取舍都用 2 秒以上的持续满载。

| 形状（M=8192，450W 持续满载） | Triton | 手写 | 提速 |
|---|---:|---:|---:|
| N=34816, K=5120 | 9.911ms | 9.177ms | 1.08× |
| N=16384, K=5120 | 4.717ms | 4.335ms | 1.09× |
| N=14336, K=5120 | 4.155ms | 3.825ms | 1.09× |
| N=5120, K=17408 | 4.990ms | 4.595ms | 1.09× |
| N=5120, K=6144 | 1.782ms | 1.661ms | 1.07× |

结果与 Triton 的平均相对差约 1e-7，来自 scale 乘积的 FP32 舍入。官方 100 请求（同 C 机，其余配置不变）：106.24/152.11/192.38/215.88 → 108.24/157.00/198.09/222.96。

### 融合 gate_up + SiLU + 量化

`down_proj` 的量化分组是 128，正好等于 tile 的 N 宽度，所以一行的 scale 在 block 结束时就已知。一个 kernel 做完 GEMM、`silu(gate)*up` 和 FP8 量化：

- 8 warps，每 warp 4 个 A 分片 × 2 gate + 2 up 分片，`g` 和 `u` 落在同一 lane，激活直接在累加器上算，用 `ex2.approx` + `rcp` 约 6 条指令。
- 每行 absmax 分三级收敛：lane 写 SMEM，分组求 max，每行一个线程算出 scale 再广播回去。借用块结束后已空闲的 A 缓冲，不额外占 SMEM。
- 单层 MLP 的激活流量从 1.86 GB 降到 0.14 GB。

| 450W 持续满载，M=8192 | 单次 |
|---|---:|
| Triton GEMM + SiLU + 量化 | 10.97ms |
| 手写 GEMM + SiLU + 量化 | 10.33–10.51ms |
| **融合 kernel** | **9.25ms（−15.7%）** |

解量化误差 2.16%，原链路 2.13%，差值是对照本身的 bf16 舍入：融合版直接用未舍入的 FP32 累加器。c1 TTFT 从手写 GEMM 的 1.733s 降到 1.655s。

### 省下的显存换成 draft 10

官方 prompt 下 draft 8 的接受率常常是 1.00，draft 是 c1/c2 最强的杠杆，每个 draft token 要 0.63 GB 的 GDN 回滚状态。显存来自三处：

| 来源 | 省出 | 做法 |
|---|---:|---|
| HTTP 父进程的 CUDA 上下文 | 454 MiB | torchao、sgl_kernel、flashinfer 导入时都查询设备，`scripts/launch_sglang_parent_off_gpu.py` 只在父进程给常量 |
| 视觉塔 | 0.92 GB | `--language-model-only`；基类已支持，只差白名单，`patches/apply_sglang_language_only.py` |
| FP8 lm_head | 1.15 GB | 见前文 |

draft 10 接受率 0.90–0.96，c1 TPOT P50 从 5.40ms 降到 4.66ms。draft 12 的 c1/c2 更高（129.64/176.40），但高并发接受率掉到 0.53–0.66，c8 只有 214.96，所以停在 10。这也修正了一个旧判断：DFlash2 的 block_size 是 8，但 draft 超过 8 依然有效。

### 少搬数据：GDN prefill 融合

profile 显示这些 kernel 全都顶着显存带宽，tile 搜索 193 组只省 4.7 ms，所以方向不是算得更快，而是少写少读。

| 改动 | 省下的流量 | 单层耗时 |
|---|---|---:|
| conv 直接写 q、k、v 三段连续缓冲区 | 去掉 3 次 `.contiguous()`，共 144 次、17.1 ms。根因是 qkv_dim 10240 超过 `MAX_FUSED_QKV_SPLIT_DIM=8192` | 0.747→0.379 ms |
| conv 顺带归一化 q、k | 省两次 l2norm 的读写 | 含在上一行之后的合计里 |
| `o` 并进 h kernel | h 和 v_new 不再写回显存再读出 | 1.186→0.676 ms |
| 三项合计（完整 GDN prefill） | | 3.07→2.165 ms |
| 输出 norm 按 128 组直接量化 | BF16 输出不再写回再读出，一行正好是一组 | 0.501→0.283 ms |

一次 8192 token prefill 的 kernel 时间从 1582 ms 降到 1532 ms，配对 TTFT 从 1625 ms 起继续下降，正式跑分 c1 TTFT 1.569s。`patches/apply_sglang_gdn_prefill.py`。

### 投机验证不再写完整状态

验证时每个 draft token 都存一份完整 GDN 状态用于回滚：8 路时每个 draft token 占 0.63 GB，c8 每步约 6 GB 写入，占一步的 10%。`--enable-linear-replayssm-spec` 换成紧凑的 (d, k, g) 窗口，提交时把接受的部分折叠进状态。SGLang 只给 MTP 接了这条路，DFlash 直接调用只懂 KDA 的提交函数，于是在配置检查里拒绝 GDN 模型。DFlash 手里已有通用提交需要的全部索引，`patches/apply_sglang_dflash_replayssm.py` 把它接上。

结果：KV 从 123585 涨到 283423 token，同机同请求 draft 10 的 c8 +2.0%，c1 decode 0.976×，也让 draft 13 不再受显存约束。

### draft 13 + FP8 草稿

权重改成 tile 优先后，draft 又往上推了一档。同机 100 请求：draft 10 = 123.62/177.74/219.38/247.39，draft 12 = 133.66/179.10/225.32/239.57，draft 13 = 137.65/182.99/220.39/243.98。draft 16 的 c1/c2/c4 更高（148.23/197.32/233.08），但 c8 掉到 234.98：M=136 超过了 decode tile 覆盖的 128，走了 prefill 配置。没有继续调，留作后续空间。

## 6. 失败案例

| 尝试 | 结果 | 原因 |
|---|---|---|
| vLLM 动态投机深度 | c1 TPOT 44.59ms，reject | 自动把图模式降为 PIECEWISE |
| vLLM MTP3 未指定草稿后端 | TPOT 32.55ms | 0.25 的草稿后端要在 `speculative_config` 里单独写，否则回退 FlashInfer 并静默关图 |
| L40S 配置直接套用 | 未能证明有效 | 只做了跨版本比较，后被同进程实测取代 |
| DFlash2（vLLM 0.25.1，draft 4） | c1 32.55 tok/s；draft 4 + 8 路 OOM | 随机 token-ID 输入下接受率仅 0.20–0.39；该结论只适用于 vLLM 0.25.1。SGLang 0.5.22 + 官方文本输入 + draft 8 重测后反超 MTP4，见 §3 |
| MTP7 | c8 48.91 tok/s | TTFT 71s，并发吞吐崩溃 |
| chunk 4096 / 16384、关 prefill 图、SSM fp16、adaptive | 无收益或变差 | 见账本 |
| 草稿模型在线 FP8（早期） | 同机 20 请求 c1 +3%，c4/c8 −2~−3%，KV 108640 | 当时接受率降到 0.83，抵消了草稿加速；tile 优先后 draft 可以加深，已转为正收益并进入提交配置 |
| 自定义块级 FP8 草稿 | u13 136.54/189.55/217.97/238.59，自带 FP8 的 t13q 135.33/181.29/208.81/242.06，c1 打平 | 没有稳定优势，代码已删 |
| FlashInfer GDN prefill | 不可用 | 要求 SM90+，4090 是 sm_89 |
| 官方数据集上的前缀缓存 | 无命中 | prompt 是随机词，两两公共前缀 0–1 字符 |
| 开回 prefill 图（FP8 草稿，mem 0.90） | 启动 OOM | prefill 图显存开销大于腾出的空间 |
| 显存/核心时钟 | `-ac`、`-cc` 返回成功但负载下仍 P2 10251 MHz，`-lmc`/`-lgc` 无权限 | 容器内不可调，功耗墙 450W 也是上限 |
| GDN prefill 分段（2048 一段，状态接力） | 数值等价（4e-9），瞬时显存 1.61→0.91 GB，但 c1 TTFT 1.655→1.725s | draft 10 不缺显存，时间代价不划算，补丁已删 |
| chunk 4096 | c1 TTFT 1.708s，对 8192 的 1.655s 变差 | 激活变小的收益抵不过多一轮前向 |
| `no_buffer` Mamba 策略 + 24 槽 | 122.17/167.57/207.50/237.84，同卡 extra_buffer 123.20/170.49/194.13/238.34 | 无整体优势，还要放弃 overlap 调度 |
| draft 12（早期） | c1/c2 更高，c8 降到 214.96，接受率 0.53–0.66 | 旧 tile 下高并发接受率崩；现已被 draft 13 取代 |
| GEMM K 循环精确改写 | 接入后 M=8192 仅 1.01–1.02×，decode 约 1.00× | 独立原型的 1.04–1.07× 未复现；低于噪声，未并入 |
| K 循环内只乘一次 scale | 1.09–1.12×，需改量化分组 | 收益小且要动权重与激活量化，reject |
| 窗口 512/1024 | c8 −1~−3% | 上下文太短 |
| A 机显卡故障 | `cuInit` 返回 NO_DEVICE | `[GPU requires reset]`，重启无效，stop/start 换卡后恢复 |

## 7. 未解决的问题

- **358 TFLOPS 不是可用上限的全部含义**：这是寄存器驻留的 mma 峰值（约 2750 MHz、259W）。真实负载受 450W 约束，手写 GEMM 在 2655 MHz 下达到 318 TFLOPS，占用率约 92%。余下的空间要靠降低每 MAC 能耗，而不是再提占用率。
- **旧的手写 PTX 峰值（2 万+ TMAC/s）已作废**：测试没有回写累加器，mma 循环被 ptxas 整体删除，测到的只是空 kernel 启动开销。修正见 `logs/artifacts/20261030_48g/C/infer/peak2.py`。
- **prefill 的瓶颈是 FP8 GEMM（81%）**：GDN 融合后其余部分只剩 294 ms，最大的是 attention 89 ms 和 add_rmsnorm 42 ms。
- **tile 网格仍不是全 M 覆盖**：五形状已补 9/72，其余 M 按最近值复用；lm_head 形状 (248320, 5120) 单独调过，增益 1.02–1.09×。
- **lm_head 仍走 Triton**：转 tile 优先预计每步可省约 0.2 ms，未做。
- **prefill 还剩 attention（约 5.8%）没有手写**：FA2 风格的 attention 改写工作量大。add_rmsnorm 与量化融合预计省约 13 ms，同样未做。
- **样本量与卡差会误导判断**：20 请求下 FP8 lm_head 的 c1 是 112.78，100 请求是 106.24；DSpark 的 c8 从 187.67 变成 202.53。B、C 两张同型号卡的 TTFT 也差约 4%（1.90s 对 1.82s）。所以取舍一律用同机 100 请求。
- **官方文档性能表未注明是否开 MTP**，因此不宣称相对官方表的提升比例，只比较我们自己的调优前后。
- **文本与 token-ID 输入差异**：官方工具默认模式的 MTP 接受率高于随机 token-ID，两种数字不可混比。
- **SGLang 0.5.22 在 sm89 上不在官方验证矩阵内**，flashinfer-cubin 版本检查需要绕过。
