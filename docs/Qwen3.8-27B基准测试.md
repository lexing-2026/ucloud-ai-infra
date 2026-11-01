# Qwen3.8-27B基准测试

## 性能测试

机器信息：48GB VRAM RTX 4090 * 1

服务启动命令（仅供参考）：

```bash
vllm serve /model/ModelScope/Qwen/Qwen3.8-27B-FP8 \
  --host 0.0.0.0 --port 8088 \
  --tensor-parallel-size 1 --dtype auto \
  --gpu-memory-utilization 0.90 --max-model-len 32768 \
  --max-num-seqs 4 --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 --served-model-name qwen3.8-27b-fp8
  # 开启MTP
  # --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

**负载与测试方法**

- 闭环并发：始终维持指定数量的在途请求，完成一个后立即补发；未额外限制请求速率。
- 每条输入由模型 tokenizer 的有效、非 special token ID 池独立随机抽取 8192 个 ID，使用 AIPerf `single-turn` 数据集的 `extra.prompt` 传递给 completions 接口，避免随机文本重编码造成长度漂移。
- 每条数据设置 `output_length=512`；请求设置 `ignore_eos=true`、`temperature=0`，保证服务端实际生成 512 token。
- 不添加公共前缀；逐档使用不同随机种子和足够多的独立样本；正式请求均核验 ISL=8192、OSL=512。
- 每档先预热 `max(4, 并发数)` 个请求，等待完成后再正式计时；预热不进入延迟百分位和吞吐量统计。
- AIPerf 使用 2 个 worker，请求超时为 180 秒；导出汇总、逐请求指标和原始请求/响应。
- 档位依次执行 1、2、4、8、3，最后更换种子对并发 2 进行 100 请求复测；测试任务之间不重叠。
- GPU 通过独立 `nvidia-smi` 进程每秒采样；AIPerf 自带 GPU telemetry 在校准中出现心跳问题，正式测试将其关闭。

**实测结果**

| 客户端并发        | 正式请求数 | 测量时长（秒） | TTFT P95（秒） | TPOT P95（毫秒/token） | 端到端 P95（秒） |  请求/s | 输出 token/s |
| ----------------- | ---------: | -------------: | -------------: | ---------------------: | ---------------: | ------: | -----------: |
| 1                 |         12 |         234.32 |          3.481 |                 31.441 |           19.527 | 0.05121 |        26.22 |
| 2                 |         24 |         286.01 |          5.023 |                 37.909 |           23.826 | 0.08391 |        42.96 |
| 3                 |         24 |         220.13 |          7.499 |                 44.974 |           28.865 | 0.10902 |        55.82 |
| 4                 |         32 |         249.08 |          9.395 |                 51.992 |           34.914 | 0.12847 |        65.78 |
| 8                 |         24 |         186.76 |         41.822 |                 53.106 |           67.417 | 0.12850 |        65.79 |
| 2（100 请求复测） |        100 |        1191.12 |          4.608 |                 37.905 |           23.818 | 0.08395 |        42.98 |

## 精度测试

数据集（Benchmark）：IFBench （前100样本，limit=100）

|       | Qwen3.8-27B-FP8 (With MTP)      | Qwen3.8-27B-FP8 (Without MTP) | Qwen3.8-27B (With MTP) | Qwen3.8-27B (Without MTP) |
| ----- | ------------------------------- | ----------------------------- | ---------------------- | ------------------------- |
| score | encounter infinite thought loop | 80%                           | 76%                    | 76%                       |

说明：为了方便测试迭代，这里只选择了100个样本来评估，无法完全代表模型的能力，有条件的同学可测试完整的数据集，以上数据仅供参考

测评python脚本：

```python
from evalscope import run_task
from evalscope.config import TaskConfig
from evalscope.api.model import GenerateConfig

task_cfg = TaskConfig(
    model='Qwen/Qwen3.8-27B-FP8',
    api_url='https://example.com/v1',
    api_key='sk-your-secret-key',
    datasets=['ifbench'],
    eval_batch_size=2,
    limit=100,
    generation_config=GenerateConfig(
        max_tokens=32768,
    ),
)
```

