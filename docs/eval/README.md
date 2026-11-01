# 评测工具

通过的 OpenAI 兼容 API 依次运行 AIPerf 性能测试和 EvalScope IFBench 精度测试，输出中文 Markdown 与 JSON 报告。需要 Bash 和网络连接。

## 使用

```bash
# 仅检查环境；依赖不齐时退出码为 2，不安装、不请求模型
bash run.sh --mode check

# 展示四档命令及 IFBench 配置；不安装、不创建输出目录、不请求模型
bash run.sh \
  --base-url http://SERVER:8088/v1 \
  --model qwen3.8-27b-fp8 \
  --dry-run

# 正式评测；如有鉴权，先通过终端安全地设置 OPENAI_API_KEY 环境变量
bash run.sh \
  --base-url http://SERVER:8088/v1 \
  --model qwen3.8-27b-fp8 \
  --mode all
```

`--base-url` 是 API 根地址，不是 `/chat/completions`。允许根地址带代理前缀，如 `https://host/proxy/v1`；不带 `/v1` 时自动补上。禁止把凭据放在 URL 中。API key 默认读取 `OPENAI_API_KEY`，未设置时使用 `EMPTY`；也可通过 `--api-key-env MY_MODEL_KEY` 指定其他环境变量名。

`--tokenizer` 可省略，默认显式传给 AIPerf `--tokenizer builtin`，使用其内置 tiktoken（`o200k_base`），不根据模型名查找或下载 Hugging Face tokenizer。首次使用时，tiktoken 可能下载并缓存编码表。

默认模式发送文本，**输入 8192 按 tiktoken 计数**；模型实际分词可能不同，报告会另列服务端实际输入 token 的均值、最小值、最大值，不会把 tiktoken 的 token ID 当成模型 token ID。输出仍以服务端回报核验为 512 token。需要严格的模型端 ISL=8192 时，请显式传入匹配的 `--tokenizer`。

显式指定的模型 tokenizer 路径位于**评测机**，也支持 Hugging Face 标识；可用 `--tokenizer-revision COMMIT` 固定其版本。远程标识会下载 tokenizer 文件，不下载模型权重；内置 tiktoken 不接受 revision。脚本不启用 `trust_remote_code`，有访问限制的模型需事先配置 Hugging Face 凭据。

## 评测参数

| 项目 | 默认设置 |
|---|---|
| AIPerf | 0.13.0 |
| 性能并发 | 1、2、4、8，依次执行 |
| ISL / OSL | 默认输入 8192 按 tiktoken 计数，输出 512 按服务端计数；显式指定模型 tokenizer 时严格核验服务端 8192/512 |
| 正式请求 | 每档 100 个，可用 `--perf-requests` 调整 |
| 预热 | 每档 8 个，等待完成后开始正式测量 |
| 性能生成 | `temperature=0`、`ignore_eos=true` |
| 性能请求超时 | 600 秒，可用 `--request-timeout` 调整 |
| EvalScope | 1.13.0，安装 `ifbench` extra |
| 精度数据集 | 原生 `ifbench`，默认数据源与顺序，`limit=100` |
| 精度 batch size | `eval_batch_size=2`，可用 `--eval-batch-size` 调整，必须为正整数 |
| 精度生成 | **仅显式覆盖 `max_tokens=32768`** |

精度不继承性能阶段的 temperature、ignore_eos、并发、输出长度或随机种子；这些行为使用 EvalScope 固定版本的原生默认配置。精度的 API 请求并发通过 `--eval-batch-size` 设置，默认 2。保存的 `resolved_config.json` 可用于核对最终配置。

IFBench 返回四项原生分数：`prompt_level_strict`、`inst_level_strict`、`prompt_level_loose`、`inst_level_loose`。**评分指标为 `prompt_level_strict`**，表示严格检查下满足全部指令的提示词比例。`report.md` 会突出显示该项分数（0–1），其余三项仅供参考，不参与评分；四项原生结果均保留供核对。指令级分数按原生规则聚合，其原生计数可能超过 100；完整性检查以预测样本、执行摘要和提示词级有效评分均为 100 为准。不会用有效的少量评分冒充完整的 100 样本测评。

性能不设置 TTFT/TPOT 达标阈值，报告“完整/异常”表示请求、输出长度、导出与统计的完整性；显式指定模型 tokenizer 时也严格核验输入长度。默认 tiktoken 模式下，服务端输入长度与 8192 不同不会中止评测或被标为异常，但实际长度会写入报告和 JSON。TPOT 使用每请求 `(端到端延迟−TTFT)/(OSL−1)`，然后对请求取百分位。报告平均、P50、P95、P99、请求/s 和输出 token/s，预热除外。网络延迟计入结果。

性能在默认 tiktoken 模式下生成精确 8192 个 tiktoken 的随机英文词文本；显式指定模型 tokenizer 时保留精确 token-ID 合成输入。服务需要支持 completions、`ignore_eos`、`cache_salt` 和 token usage；显式模型 tokenizer 模式还需要 token-ID prompt 支持，适用于此前的 vLLM 服务。四档使用同一份确定性数据和顺序，每档传不同的 `cache_salt`；不会调用服务器的缓存重置接口。若服务静默忽略 cache_salt，单靠客户端无法确认缓存隔离，报告不会宣称缓存命中为零。另有一个相同输入口径、1 输出的接口预检查请求，使用不参与预热/正式测试的输入，不计入统计。

若服务公布的上下文上限 `<=32768`，输入提示词再加 `max_tokens=32768` 必然不够，脚本将精度阶段标为配置异常并跳过。若 API 不公布上限，实际上下文超限错误由 EvalScope 返回。脚本**不会自动降低 max_tokens**；参赛者需要配置足够的上下文窗口。仅提供 chat 接口的服务不能运行此性能测试，应明确报错。

## 依赖确认与隔离

入口先寻找可用于编排的 Python ≥3.10。评测工具统一使用 Python 3.12，各自放在：

```text
~/.cache/ai-infra-eval/envs/
├── aiperf-0.13.0/
├── evalscope-1.13.0/
└── nltk_data/
```

可通过 `--env-dir PATH` 更改。脚本只复用上述指定目录中版本匹配的工具环境，检查结果同时列出 PATH 中的全局命令，但不会改动全局安装。

缺失的 uv、Python、工具包及 IFBench 的 `emoji`、`syllapy`、`nltk` 和三项 NLTK 数据资源均会先列出安装操作，必须在终端明确输入 `y`/`yes` 才执行。资源包括 `punkt_tab`、`stopwords`、`averaged_perceptron_tagger_eng`。不使用 sudo，不修改系统 Python，不修改 shell 启动文件。不提供绕过安装确认的 `--yes` 参数；非交互环境应先在交互终端准备环境。

uv 使用 [官方安装器](https://docs.astral.sh/uv/getting-started/installation/)，安装到 `~/.local/bin`。完全没有 Python 时，入口使用已有 curl/wget 引导安装；若三者都不存在，给出手动准备说明并退出。安装本身需要联网；索引和代理可使用 uv/Hugging Face/ModelScope 的环境变量配置。IFBench 数据集会按 EvalScope 默认方式下载到用户缓存。

## 模式、输出和失败处理

```bash
# 仅精度：无需 tokenizer，也不安装 AIPerf
bash run.sh --mode accuracy --base-url http://SERVER:8088/v1 --model MODEL

# 调整精度评测的 API 请求并发为 4
bash run.sh --mode accuracy --base-url http://SERVER:8088/v1 \
  --model MODEL --eval-batch-size 4

# 仅性能；为快速验证手动缩小请求数（正式比赛建议保持默认 100）
bash run.sh --mode perf --base-url http://SERVER:8088/v1 \
  --model MODEL --perf-requests 2 --output-dir ./outputs/smoke
```

默认输出 `outputs/YYYYMMDD_HHMMSS_ffffff/`，也可通过 `--output-dir` 指定不存在的新目录。已有输出目录一律不覆盖。

```text
report.md                 # 中文汇总，含异常原因
summary.json              # 结构化统计
configuration.json        # 入口参数及固定评测口径
environment.json          # 平台、解释器及工具版本
model_info.json           # /models 回报
performance/
  dataset.jsonl           # tiktoken 文本或显式模型 tokenizer 的 token-ID 输入
  dataset.meta.json       # tokenizer、随机种子和数据 SHA-256
  c1/ c2/ c4/ c8/         # 各轮命令、日志、AIPerf 原生汇总/逐请求记录
accuracy/
  task_config.json        # 精度阶段传入配置
  resolved_config.json    # EvalScope 展开的默认配置，API key 脱敏
  native/                 # 原生配置、日志、预测、review、report
```

单档性能异常后继续其他档位；性能整体准备失败后仍尝试精度阶段。两类评测不会重叠运行。鉴权或模型发现失败时不发送评测负载。Ctrl+C / SIGTERM 会终止当前评测子进程组，保留已完成汇总及部分原生结果；报告标为中断。原生导出中的 API key 在阶段结束及中断清理时脱敏；输出目录权限为 0700，运行过程中不要共享未完成的目录。

退出码：`0` 完整或成功 dry-run；`1` 评测存在异常；`2` 参数/依赖/启动错误或 check 未通过；`130` 已中断。
