#!/usr/bin/env bash
# Bootstrap only; model evaluation and reporting live in runner.py.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly_mode=false
previous=""
for argument in "$@"; do
  case "$argument" in
    -h|--help)
      cat <<'HELP'
AI Infras 比赛评测（Linux / macOS，Bash + Python）

用法：bash run.sh --base-url http://SERVER:8088/v1 --model MODEL --mode all

  --mode all|perf|accuracy|check   默认 all；check 只检查依赖
  --tokenizer PATH_OR_HF_ID       可选；默认 builtin（tiktoken/o200k_base）
  --tokenizer-revision REV        可选，固定远程 tokenizer 的版本
  --api-key-env NAME              API key 环境变量名，默认 OPENAI_API_KEY
  --output-dir PATH               新建结果目录；默认 outputs/时间戳
  --env-dir PATH                  工具环境目录；默认用户缓存目录
  --perf-requests N               每档正式请求数，默认 100
  --eval-batch-size N             EvalScope 评测 batch size，默认 2
  --request-timeout SECONDS       性能请求超时，默认 600 秒
  --seed N                       性能输入随机种子，默认 42
  --dry-run                      展示配置和命令；不安装、不请求模型

性能：8192/512 token，默认输入按 tiktoken 计数；报告另列模型实际长度。
      并发 1/2/4/8，每档预热 8 请求，无延迟门槛。
精度：EvalScope IFBench，limit=100，eval batch size 默认 2，max_tokens=32768。
缺少依赖时必须交互确认才安装；非交互模式不会自动安装。
HELP
      exit 0 ;;
    --dry-run|--mode=check) readonly_mode=true ;;
  esac
  if [[ "$previous" == "--mode" && "$argument" == "check" ]]; then
    readonly_mode=true
  fi
  previous="$argument"
done

confirm() {
  local answer
  if [[ ! -t 0 ]]; then
    echo "缺少依赖；请在交互终端重新运行，确认安装。" >&2
    exit 2
  fi
  read -r -p "是否执行以上安装操作？[y/N] " answer
  case "$answer" in y|Y|yes|YES) ;; *) echo "已取消安装。" >&2; exit 2 ;; esac
}

python_bin=""
for candidate in python3.12 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>/dev/null; then
    python_bin="$(command -v "$candidate")"
    break
  fi
done

# With Python available, the runner can inspect dependencies without mutation.
if [[ -n "$python_bin" ]]; then
  exec "$python_bin" "$SCRIPT_DIR/runner.py" "$@"
fi

echo "缺少用于启动脚本的 Python >= 3.10。" >&2
if "$readonly_mode"; then
  echo "需要先准备 uv 和 Python 3.12；未执行安装。" >&2
  exit 2
fi

uv_bin="$(command -v uv || true)"
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin="$HOME/.local/bin/uv"
fi
if [[ -z "$uv_bin" ]]; then
  if ! command -v curl >/dev/null && ! command -v wget >/dev/null; then
    echo "同时缺少 uv、Python 和 curl/wget，无法引导安装。请先手动准备其中之一。" >&2
    exit 2
  fi
  cat >&2 <<'INSTALL'
缺少 uv。将下载 https://astral.sh/uv/install.sh 到临时文件，
使用 sh 执行，安装到 ~/.local/bin，不修改 shell 配置、不使用 sudo。
INSTALL
  confirm
  installer="$(mktemp)"
  trap 'rm -f -- "$installer"' EXIT
  if command -v curl >/dev/null; then
    curl --fail --silent --show-error --location --proto '=https' https://astral.sh/uv/install.sh -o "$installer"
  else
    wget -q https://astral.sh/uv/install.sh -O "$installer"
  fi
  UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh "$installer"
  uv_bin="$HOME/.local/bin/uv"
fi
echo "将执行：$uv_bin python install 3.12（用户目录中的受管 Python）" >&2
confirm
"$uv_bin" python install 3.12
python_bin="$("$uv_bin" python find --offline --no-python-downloads 3.12)"
exec "$python_bin" "$SCRIPT_DIR/runner.py" "$@"
