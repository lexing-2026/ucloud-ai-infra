#!/usr/bin/env python3
"""Remote API judging: isolated tools, exact lengths, native scores, auditable output."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid

SCRIPT = Path(__file__).resolve()
CONCURRENCIES = (1, 2, 4, 8)
ISL, OSL, WARMUP, LIMIT, MAX_TOKENS = 8192, 512, 8, 100, 32768
PACKAGES = {"perf": ("aiperf", "0.13.0"), "accuracy": ("evalscope[ifbench]", "1.13.0")}
NLTK_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab/english/",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng/",
}
IF_METRICS = ("prompt_level_strict", "inst_level_strict", "prompt_level_loose", "inst_level_loose")


class EvaluationError(Exception):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def redact(text: str, secret: str) -> str:
    if secret and secret != "EMPTY":
        for value in {secret, json.dumps(secret)[1:-1], urllib.parse.quote(secret, safe="")}:
            text = text.replace(value, "<REDACTED>")
    return text


def write_json(path: Path, data: object, secret: str = "") -> None:
    path.write_text(redact(json.dumps(data, ensure_ascii=False, indent=2), secret) + "\n", encoding="utf-8")


def sanitize_files(folder: Path, secret: str) -> None:
    """Native tool exports can contain credentials even when our command log does not."""
    if not secret or secret == "EMPTY":
        return
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".log", ".yaml", ".yml", ".txt", ".csv", ".md"}:
            continue
        temporary = path.with_name(path.name + ".redacted")
        try:
            with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as dest:
                for line in source:
                    dest.write(redact(line, secret))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def execute(command: list[str], log: Path | None = None, secret: str = "", env: dict | None = None) -> None:
    print("执行：" + redact(shlex.join(command), secret), flush=True)
    stream = log.open("w", encoding="utf-8") if log else None
    child = None
    try:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", env=env, start_new_session=True)
        for line in child.stdout:
            safe_line = redact(line, secret)
            if stream:
                stream.write(safe_line)
                stream.flush()
            else:
                print(safe_line, end="", flush=True)
        code = child.wait()
        if code:
            raise EvaluationError(f"子进程退出码 {code}；日志：{log or '终端输出'}")
    finally:
        if child and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if child:
            # Clean up any remaining workers after their launcher has exited.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            if child.stdout:
                child.stdout.close()
        if stream:
            stream.close()


def confirm(message: str, commands: list[list[str]]) -> None:
    print(message, flush=True)
    for command in commands:
        print("  " + shlex.join(command))
    if not sys.stdin.isatty():
        raise EvaluationError("非交互环境不安装依赖。请在交互终端重新运行并确认以上操作。")
    if input("是否执行以上安装操作？[y/N] ").strip().lower() not in {"y", "yes"}:
        raise EvaluationError("用户未同意安装，已退出。")


def find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    local = Path.home() / ".local/bin/uv"
    return str(local) if local.is_file() and os.access(local, os.X_OK) else None


def find_python312() -> str | None:
    candidates = [shutil.which("python3.12")]
    if sys.version_info[:2] == (3, 12):
        candidates.append(sys.executable)
    managed = Path(os.environ.get("UV_PYTHON_INSTALL_DIR", Path.home() / ".local/share/uv/python"))
    candidates.extend(str(p) for p in sorted(managed.glob("cpython-3.12.*/bin/python3.12")))
    for candidate in candidates:
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    return None


def tool_python(env_dir: Path, stage: str) -> Path:
    package, version = PACKAGES[stage]
    return env_dir / f"{package.split('[')[0]}-{version}" / "bin/python"


def inspect_tool(python: Path, stage: str, nltk_dir: Path) -> dict:
    if not python.is_file():
        return {"ready": False, "missing": ["独立虚拟环境"], "python": str(python)}
    package, version = PACKAGES[stage]
    modules = [package.split("[")[0]]
    if stage == "perf":
        modules.extend(["transformers", "tiktoken"])
    else:
        modules.extend(["emoji", "syllapy", "nltk"])
    code = """
import importlib.metadata as m, importlib.util, json, sys
modules, expected, resources = json.loads(sys.argv[1])
missing = [name for name in modules if importlib.util.find_spec(name) is None]
try: version = m.version(modules[0])
except m.PackageNotFoundError: version = None
if version != expected: missing.append(f'{modules[0]}=={expected} (installed={version})')
if 'nltk' in modules and 'nltk' not in missing:
 import nltk
 for name, location in resources.items():
  try: nltk.data.find(location)
  except LookupError: missing.append('nltk:' + name)
print(json.dumps({'ready': not missing, 'missing': missing, 'version': version,
                  'python_version': list(sys.version_info[:3]),
                  'packages': {d.metadata['Name']: d.version for d in m.distributions()}}))
"""
    result = subprocess.run([str(python), "-c", code, json.dumps([modules, version, NLTK_RESOURCES if stage == "accuracy" else {}])],
                            text=True, capture_output=True, env=child_environment(nltk_dir))
    if result.returncode:
        return {"ready": False, "missing": ["环境不可用；请检查解释器及其依赖"], "python": str(python)}
    data = json.loads(result.stdout)
    data["python"] = str(python)
    if data["python_version"][:2] != [3, 12]:
        data["ready"] = False
        data["missing"].append("比赛统一 Python 3.12；请改用新的 --env-dir")
    return data


def child_environment(nltk_dir: Path) -> dict:
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NLTK_DATA"] = str(nltk_dir) + (os.pathsep + env["NLTK_DATA"] if env.get("NLTK_DATA") else "")
    return env


def dependency_status(args: argparse.Namespace, stages: list[str]) -> dict:
    return {
        "uv": find_uv(), "bootstrap_python": sys.executable, "python312": find_python312(),
        "tools": {stage: inspect_tool(tool_python(args.env_dir, stage), stage, args.env_dir / "nltk_data") for stage in stages},
        "global_commands": {name: shutil.which(name) for name in ("aiperf", "evalscope")},
    }


def ensure_dependencies(args: argparse.Namespace, stages: list[str]) -> dict:
    status = dependency_status(args, stages)
    if not status["uv"]:
        destination = Path.home() / ".local/bin"
        confirm(f"缺少 uv。将下载官方安装器到临时文件，安装到 {destination}，不修改 shell 配置。", [["sh", "<https://astral.sh/uv/install.sh>"]])
        with tempfile.TemporaryDirectory(prefix="ai-infra-uv-") as temporary:
            installer = Path(temporary) / "install.sh"
            with urllib.request.urlopen("https://astral.sh/uv/install.sh", timeout=60) as response:
                installer.write_bytes(response.read())
            env = {**os.environ, "UV_INSTALL_DIR": str(destination), "UV_NO_MODIFY_PATH": "1"}
            execute(["sh", str(installer)], env=env)
        status["uv"] = find_uv()
        if not status["uv"]:
            raise EvaluationError("uv 安装后仍未找到可执行文件。")
    uv = status["uv"]
    commands = []
    if not status["python312"]:
        commands.append([uv, "python", "install", "3.12"])
    missing_stages = [stage for stage in stages if not status["tools"][stage]["ready"]]
    for stage in missing_stages:
        python = tool_python(args.env_dir, stage)
        if python.exists() and status["tools"][stage].get("python_version", [3, 12])[:2] != [3, 12]:
            raise EvaluationError(f"{python} 不是 Python 3.12；请使用新的 --env-dir，避免覆盖已有环境。")
        if not python.exists():
            commands.append([uv, "venv", "--no-project", "--python", "3.12", "--no-python-downloads", str(python.parent.parent)])
        package, version = PACKAGES[stage]
        commands.append([uv, "pip", "install", "--python", str(python), f"{package}=={version}"])
        if stage == "accuracy":
            commands.append([str(python), "-m", "nltk.downloader", "-d", str(args.env_dir / "nltk_data"), *NLTK_RESOURCES])
    if commands:
        details = {stage: status["tools"][stage]["missing"] for stage in missing_stages}
        confirm(f"需要准备隔离评测环境：{args.env_dir}\n缺失项：{json.dumps(details, ensure_ascii=False)}", commands)
        for command in commands:
            execute(command, env=child_environment(args.env_dir / "nltk_data"))
    status = dependency_status(args, stages)
    if any(not status["tools"][stage]["ready"] for stage in stages):
        raise EvaluationError("依赖安装或资源下载后检查仍未通过：" + json.dumps(status, ensure_ascii=False))
    return status


def normalize_api_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("URL 必须是 HTTP(S) API 根地址，不能包含凭据、查询参数或片段。")
    path = parsed.path.rstrip("/")
    if path.endswith(("/completions", "/models", "/responses")):
        raise argparse.ArgumentTypeError("请提供 API 根地址（例如 http://host:8088/v1），不要提供具体接口。")
    if not path.endswith("/v1"):
        path += "/v1"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def http_json(url: str, key: str, payload: dict | None = None, timeout: float = 15, streaming: bool = False) -> dict:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if streaming:
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    raise EvaluationError("接口未返回 SSE 流式响应。")
                usage = None
                done = False
                for line in response:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(body)
                    if chunk.get("error"):
                        raise EvaluationError(redact(f"流式 API 返回错误：{chunk['error']}", key))
                    usage = chunk.get("usage") or usage
                if not done or not usage:
                    raise EvaluationError("流式响应缺少 [DONE] 或 token usage。")
                return {"usage": usage}
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(8192).decode(errors="replace")
        raise EvaluationError(redact(f"API HTTP {error.code}: {detail}", key)) from error
    except (OSError, ValueError) as error:
        raise EvaluationError(redact(f"API 请求失败：{error}", key)) from error


def check_model(base_url: str, model: str, key: str) -> dict:
    response = http_json(base_url + "/models", key)
    matching = [item for item in response.get("data", []) if item.get("id") == model]
    if not matching:
        raise EvaluationError(f"/models 中未找到指定模型 {model!r}。")
    return matching[0]


def dataset_worker(tokenizer_id: str, revision: str | None, target: Path, count: int, seed: int) -> None:
    if tokenizer_id == "builtin":
        from aiperf.common.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_pretrained("builtin")
        # Single-token ASCII words avoid invalid UTF-8 boundaries and foreign model IDs.
        words = ("apple book city cloud code data day earth field fire forest game gold green home house idea king "
                 "lake land light line list map moon name night north ocean paper park path plan rain red river road "
                 "rock room sand sea seed sky snow star stone story sun team test time town tree water way wind "
                 "word work world year").split()
        pool = [word for word in words if len(tokenizer.encode(word)) == 1 and len(tokenizer.encode(" " + word)) == 1]
        commit = None
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=revision, trust_remote_code=False)
        pool = sorted(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids))
        commit = tokenizer.init_kwargs.get("_commit_hash")
    if not pool:
        raise EvaluationError("tokenizer 没有可用的合成输入词元。")
    rng = random.Random(seed)
    with target.open("w", encoding="utf-8") as stream:
        for _ in range(count):
            if tokenizer_id == "builtin":
                text = " ".join(rng.choices(pool, k=ISL))
                if len(tokenizer.encode(text)) != ISL:
                    raise EvaluationError("内置 tiktoken 输入长度核验失败。")
                record = {"text": text, "output_length": OSL}
            else:
                record = {"text": "Synthetic token-ID prompt; see extra.prompt", "output_length": OSL,
                          "extra": {"prompt": rng.choices(pool, k=ISL)}}
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    write_json(target.with_suffix(".meta.json"), {
        "tokenizer": tokenizer_id, "revision": revision, "seed": seed, "entries": count,
        "pool_size": len(pool), "isl": ISL, "osl": OSL,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_commit": commit,
        "prompt_format": "text" if tokenizer_id == "builtin" else "token_ids",
        "input_length_basis": "tiktoken/o200k_base" if tokenizer_id == "builtin" else "model",
    })


def accuracy_worker(config_path: Path) -> None:
    from evalscope import run_task
    from evalscope.config import TaskConfig
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    key = os.environ.get(spec.pop("api_key_env"), "EMPTY")
    config = TaskConfig(**spec, api_key=key)
    resolved = config.model_dump(mode="json")
    write_json(config_path.with_name("resolved_config.json"), resolved, key)
    run_task(task_cfg=config)


def aiperf_command(args: argparse.Namespace, concurrency: int, dataset: Path, dest: Path, salt: str, key: str) -> list[str]:
    api = urllib.parse.urlsplit(args.base_url)
    origin = urllib.parse.urlunsplit((api.scheme, api.netloc, "", "", ""))
    command = [str(tool_python(args.env_dir, "perf").with_name("aiperf")), "profile", "--model", args.model,
               "--url", origin, "--endpoint", api.path + "/completions", "--endpoint-type", "completions", "--streaming",
               "--tokenizer", args.tokenizer, "--use-server-token-count", "--use-legacy-max-tokens",
               "--input-file", str(dataset), "--custom-dataset-type", "single-turn",
               "--dataset-sampling-strategy", "sequential", "--random-seed", str(args.seed),
               "--extra-inputs", json.dumps({"ignore_eos": True, "temperature": 0, "cache_salt": salt}),
               "--concurrency", str(concurrency), "--request-count", str(args.perf_requests),
               "--warmup-request-count", str(WARMUP), "--workers-max", "2",
               "--request-timeout-seconds", str(args.request_timeout), "--export-level", "records",
               "--artifact-dir", str(dest), "--ui", "none", "--no-gpu-telemetry", "--api-key", key]
    if args.tokenizer_revision:
        command.extend(["--tokenizer-revision", args.tokenizer_revision])
    return command


def accuracy_config(args: argparse.Namespace, dest: Path) -> dict:
    return {"model": args.model, "api_url": args.base_url, "api_key_env": args.api_key_env,
            "eval_type": "openai_api", "datasets": ["ifbench"], "limit": LIMIT,
            "eval_batch_size": args.eval_batch_size,
            "generation_config": {"max_tokens": MAX_TOKENS},
            "work_dir": str(dest / "native"), "no_timestamp": True}


def percentile(values: list[float], percent: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def distribution(values: list[float]) -> dict:
    return {"mean": sum(values) / len(values), "p50": percentile(values, 50),
            "p95": percentile(values, 95), "p99": percentile(values, 99)}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_performance(folder: Path, expected: int, concurrency: int, warmup_expected: int | None = None,
                      strict_isl: bool = True) -> dict:
    native = json.loads((folder / "profile_export_aiperf.json").read_text(encoding="utf-8"))
    records = read_jsonl(folder / "profile_export.jsonl")
    warmups = [r for r in records if r.get("metadata", {}).get("benchmark_phase") == "warmup"]
    records = [record for record in records if record.get("metadata", {}).get("benchmark_phase") == "profiling"]
    tags = ("time_to_first_token", "inter_token_latency", "request_latency", "input_sequence_length", "output_sequence_length")
    usable = [r for r in records if all(isinstance(r.get("metrics", {}).get(tag, {}).get("value"), (int, float))
                                      and math.isfinite(r["metrics"][tag]["value"])
                                      and r["metrics"][tag]["value"] >= 0 for tag in tags)]
    issues = []
    if warmup_expected is not None and (len(warmups) != warmup_expected or any(
            (strict_isl and r.get("metrics", {}).get("input_sequence_length", {}).get("value") != ISL) or
            r.get("metrics", {}).get("output_sequence_length", {}).get("value") != OSL for r in warmups)):
        issues.append(f"未完成 {warmup_expected} 个正确长度的预热请求")
    if not native.get("is_complete") or native.get("was_cancelled"):
        issues.append("AIPerf 未完整结束")
    if len(records) != expected or len(usable) != expected:
        issues.append(f"正式记录 {len(records)}、有效记录 {len(usable)}，期望 {expected}")
    successes = native.get("request_count", {}).get("avg")
    errors = native.get("error_request_count", {}).get("avg", 0)
    if successes != expected or errors or native.get("error_summary") or native.get("request_error_rate", {}).get("avg", 0):
        issues.append("存在请求失败、缺失或工具错误")
    inputs_ok = bool(usable) and all(r["metrics"]["input_sequence_length"]["value"] == ISL for r in usable)
    outputs_ok = bool(usable) and all(r["metrics"]["output_sequence_length"]["value"] == OSL for r in usable)
    if not outputs_ok or (strict_isl and not inputs_ok):
        issues.append("服务端实际 token 数不符合 8192/512" if strict_isl else "服务端实际输出 token 数不是 512")
    if any(r["metrics"]["input_sequence_length"]["value"] <= 0 for r in usable):
        issues.append("服务端实际输入 token 数无效")
    stats = {}
    for tag in tags[:3]:
        # Check the source units instead of silently comparing seconds and milliseconds.
        if any(r["metrics"][tag].get("unit") != "ms" for r in usable):
            issues.append(f"{tag} 单位不是 ms")
        values = [r["metrics"][tag]["value"] for r in usable]
        if values:
            stats[tag] = distribution(values)
            for quantile in ("p50", "p95", "p99"):
                exported = native.get(tag, {}).get(quantile)
                if exported is None or not math.isclose(exported, stats[tag][quantile], rel_tol=1e-6, abs_tol=1e-5):
                    issues.append(f"{tag} {quantile} 与原生汇总不一致")
    for r in usable:
        m = r["metrics"]
        if m["output_sequence_length"]["value"] > 1:
            derived = (m["request_latency"]["value"] - m["time_to_first_token"]["value"]) / (m["output_sequence_length"]["value"] - 1)
            if not math.isclose(derived, m["inter_token_latency"]["value"], rel_tol=1e-6, abs_tol=1e-5):
                issues.append("逐请求 TPOT 公式不一致")
                break
    scalar_tags = ("benchmark_duration", "request_throughput", "output_token_throughput")
    scalars = {tag: native.get(tag, {}).get("avg") for tag in scalar_tags}
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in scalars.values()):
        issues.append("吞吐量或测量时长缺失/无效")
    input_lengths = [r["metrics"]["input_sequence_length"]["value"] for r in usable]
    actual_inputs = ({"min": min(input_lengths), "max": max(input_lengths), "mean": sum(input_lengths)/len(input_lengths)}
                     if input_lengths else {})
    return {"status": "异常" if issues else "完整", "issues": issues, "concurrency": concurrency,
            "expected_requests": expected, "records": len(records), "valid_records": len(usable),
            "successful_requests": successes, "error_requests": errors,
            "lengths_verified": inputs_ok and outputs_ok, "osl_verified": outputs_ok,
            "input_length_basis": "model" if strict_isl else "tiktoken/o200k_base",
            "actual_input_tokens": actual_inputs, "metrics": stats, **scalars,
            "artifacts": str(folder)}


def parse_accuracy(folder: Path) -> dict:
    candidates = []
    for path in (folder / "native/reports").rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("dataset_name") == "ifbench":
            candidates.append((path, data))
    if len(candidates) != 1:
        raise EvaluationError(f"期望唯一 IFBench 原生报告，实际找到 {len(candidates)} 个。")
    path, native = candidates[0]
    metrics = {}
    for metric in native.get("metrics", []):
        name = metric.get("legacy_name") or metric.get("name") or metric.get("identity", {}).get("name")
        if name in IF_METRICS:
            if name in metrics:
                raise EvaluationError(f"IFBench 报告指标重复：{name}")
            metrics[name] = {"score": metric.get("score"), "num": metric.get("num")}
    predictions = []
    for file in (folder / "native/predictions").rglob("*.jsonl"):
        predictions.extend(read_jsonl(file))
    execution = native.get("execution_summary") or {}
    issues = []
    if len(predictions) != LIMIT:
        issues.append(f"预测记录数 {len(predictions)}，期望 {LIMIT}")
    if any(p.get("error") for p in predictions):
        issues.append("预测记录包含推理错误")
    if any(not p.get("model_output", {}).get("choices") for p in predictions if p.get("model_output") is not None):
        issues.append("预测记录缺少模型输出 choices")
    if any(p.get("model_output") is None for p in predictions):
        issues.append("预测记录缺少 model_output")
    indices = [p.get("index") for p in predictions]
    if any(index is None for index in indices) or len(set(indices)) != len(indices):
        issues.append("预测样本 index 缺失或重复")
    for name in IF_METRICS:
        metric = metrics.get(name, {})
        score = metric.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            issues.append(f"{name} 缺少有效原生分数")
    # Instruction metrics use instruction-weighted counts; they need not equal 100.
    for name in ("prompt_level_strict", "prompt_level_loose"):
        if metrics.get(name, {}).get("num") != LIMIT:
            issues.append(f"{name} 有效评分样本不是 {LIMIT}")
    if execution and (execution.get("requested") != LIMIT or execution.get("succeeded") != LIMIT or
                      execution.get("errored") or execution.get("incomplete")):
        issues.append("原生 execution_summary 显示评测不完整")
    if not execution:
        issues.append("缺少原生 execution_summary，不能核验完整性")
    return {"status": "异常" if issues else "完整", "issues": issues, "metrics": metrics,
            "generated_samples": len(predictions), "execution_summary": execution,
            "native_report": str(path), "artifacts": str(folder)}


def probe_performance(args: argparse.Namespace, dataset: Path, key: str, salt: str) -> None:
    # The spare final prompt is never used by warmup or profiling.
    last = None
    with dataset.open(encoding="utf-8") as stream:
        for line in stream:
            last = line
    record = json.loads(last)
    prompt = record.get("extra", {}).get("prompt", record["text"])
    response = http_json(args.base_url + "/completions", key,
                         {"model": args.model, "prompt": prompt, "max_tokens": 1, "ignore_eos": True,
                          "temperature": 0, "cache_salt": salt, "stream": True,
                          "stream_options": {"include_usage": True}}, args.request_timeout, streaming=True)
    usage = response.get("usage", {})
    actual_isl = usage.get("prompt_tokens")
    if (not isinstance(actual_isl, int) or actual_isl <= 0 or usage.get("completion_tokens") != 1 or
            (args.tokenizer != "builtin" and actual_isl != ISL)):
        raise EvaluationError("性能接口预检查失败：需要正确处理 prompt、ignore_eos、cache_salt 并返回准确 usage。")


def run_performance(args: argparse.Namespace, root: Path, key: str, run_id: str) -> list[dict]:
    folder = root / "performance"
    folder.mkdir()
    if args.tokenizer == "builtin":
        print("使用内置 tiktoken/o200k_base：输入目标 8192 按 tiktoken 计数；服务端实际输入长度会单独列入报告。", flush=True)
    dataset = folder / "dataset.jsonl"
    python = tool_python(args.env_dir, "perf")
    execute([str(python), str(SCRIPT), "_dataset", args.tokenizer, args.tokenizer_revision or "",
             str(dataset), str(args.perf_requests + WARMUP + 1), str(args.seed)],
            folder / "dataset.log", key, child_environment(args.env_dir / "nltk_data"))
    probe_performance(args, dataset, key, run_id + ":probe")
    results = []
    for concurrency in CONCURRENCIES:
        dest = folder / f"c{concurrency}"
        dest.mkdir()
        command = aiperf_command(args, concurrency, dataset, dest, f"{run_id}:c{concurrency}", key)
        write_json(dest / "command.json", command, key)
        print(f"\n性能测试：并发 {concurrency}，正式 {args.perf_requests}，预热 {WARMUP}", flush=True)
        failure = None
        try:
            execute(command, dest / "console.log", key, child_environment(args.env_dir / "nltk_data"))
        except EvaluationError as error:
            failure = str(error)
        finally:
            sanitize_files(dest, key)
        try:
            result = parse_performance(dest, args.perf_requests, concurrency, warmup_expected=WARMUP,
                                       strict_isl=args.tokenizer != "builtin")
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = {"status": "异常", "issues": [f"结果解析失败：{error}"], "concurrency": concurrency, "artifacts": str(dest)}
        if failure:
            result["status"] = "异常"
            result["issues"].append(failure)
        results.append(result)
        write_json(dest / "summary.json", result, key)
        # Checkpoint each completed cell, including before a later interruption.
        write_json(folder / "summary.json", results, key)
        print(f"并发 {concurrency}：{result['status']}；{'; '.join(result['issues']) or '长度和统计核验完成'}", flush=True)
    return results


def run_accuracy(args: argparse.Namespace, root: Path, key: str, model_info: dict) -> dict:
    folder = root / "accuracy"
    folder.mkdir()
    config = accuracy_config(args, folder)
    write_json(folder / "task_config.json", config)
    length = model_info.get("max_model_len") or model_info.get("context_length")
    if isinstance(length, (int, float)) and length <= MAX_TOKENS:
        result = {"status": "异常", "issues": [f"模型上下文上限 {length} 无法容纳 IFBench 提示词 + max_tokens={MAX_TOKENS}；未自动缩短输出。"], "artifacts": str(folder)}
        write_json(folder / "summary.json", result)
        return result
    command = [str(tool_python(args.env_dir, "accuracy")), str(SCRIPT), "_accuracy", str(folder / "task_config.json")]
    write_json(folder / "command.json", command)
    failure = None
    try:
        execute(command, folder / "console.log", key, child_environment(args.env_dir / "nltk_data"))
    except EvaluationError as error:
        failure = str(error)
    finally:
        sanitize_files(folder, key)
    try:
        result = parse_accuracy(folder)
    except (OSError, ValueError, KeyError, TypeError, EvaluationError) as error:
        result = {"status": "异常", "issues": [str(error)], "artifacts": str(folder)}
    if failure:
        result["status"] = "异常"
        result["issues"].append(failure)
    write_json(folder / "summary.json", result, key)
    return result


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def metric_cell(row: dict, metric: str, quantile: str, factor: float = 1) -> str:
    value = row.get("metrics", {}).get(metric, {}).get(quantile)
    return f"{value * factor:.3f}" if isinstance(value, (int, float)) else "—"


def number_cell(value: object, digits: int) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def render_report(root: Path, summary: dict, secret: str = "") -> None:
    args = summary["configuration"]
    shape = "ISL/OSL=8192/512（服务端口径）"
    if args.get("tokenizer", "builtin") == "builtin":
        shape = "输入目标 8192 按内置 tiktoken/o200k_base 计数；OSL=512 按服务端计数"
    lines = ["# AI Infras 比赛评测报告", "", f"- 模型：`{md_cell(args['model'])}`",
             f"- API：`{md_cell(args['base_url'])}`", f"- 开始：{summary['started_at']}",
             f"- 结束：{summary.get('ended_at', '未完成')}", f"- 总体状态：**{summary['status']}**", "",
             "## 性能测试", "", shape + "；并发 1/2/4/8；每档预热 8 请求，不计入正式统计。",
             "延迟包含评测机到 API 的网络开销，不设置 TTFT/TPOT 达标门槛。", "",
             "| 并发 | 状态 | 成功请求 | 错误请求 | 请求/s | 输出 token/s |",
             "|---:|---|---:|---:|---:|---:|"]
    perf = summary.get("performance", [])
    for row in perf:
        lines.append(f"| {row['concurrency']} | {row['status']} | {row.get('successful_requests', '—')} | {row.get('error_requests', '—')} | "
                     f"{number_cell(row.get('request_throughput'), 5)} | {number_cell(row.get('output_token_throughput'), 2)} |")
    lines += ["", "| 并发 | 服务端实际输入 token 均值 | 最小 | 最大 |", "|---:|---:|---:|---:|"]
    for row in perf:
        lengths = row.get("actual_input_tokens", {})
        lines.append(f"| {row['concurrency']} | {number_cell(lengths.get('mean'), 1)} | "
                     f"{number_cell(lengths.get('min'), 0)} | {number_cell(lengths.get('max'), 0)} |")
    if args.get("tokenizer", "builtin") == "builtin":
        lines += ["", "tiktoken 与模型分词可能不同，因此服务端实际 ISL 不保证等于 8192；上表使用服务端 usage。",
                  "完整状态表示请求、输出长度和统计完整，不表示模型端 ISL 恰为 8192。"]
    lines += ["", "| 并发 | 指标 | 平均 | P50 | P95 | P99 |", "|---:|---|---:|---:|---:|---:|"]
    for row in perf:
        for tag, label, factor in [("time_to_first_token", "TTFT（秒）", .001),
                                   ("inter_token_latency", "TPOT（毫秒/token）", 1),
                                   ("request_latency", "端到端（秒）", .001)]:
            lines.append(f"| {row['concurrency']} | {label} | " + " | ".join(metric_cell(row, tag, q, factor) for q in ("mean", "p50", "p95", "p99")) + " |")
    lines += ["", "TPOT 为每请求 `(端到端延迟 − TTFT)/(输出 token 数 − 1)` 后取百分位；不使用逐块间隔的 P95。",
              "使用相同确定性输入及顺序，每轮请求传独立 cache_salt 避免跨档缓存复用；未通过服务器遥测独立验证缓存命中。", "",
              "## IFBench 精度", "", f"EvalScope 原生评测，limit=100，eval_batch_size={args.get('eval_batch_size', 2)}；"
              "生成参数仅覆盖 `max_tokens=32768`，其余沿用工具默认值。"]
    accuracy = summary.get("accuracy")
    if accuracy:
        primary_score = accuracy.get("metrics", {}).get("prompt_level_strict", {}).get("score", "—")
        lines += [f"状态：**{accuracy['status']}**；预测记录数：{accuracy.get('generated_samples', '—')}。", "",
                  f"**评分指标：`prompt_level_strict`；评分分数（0–1）：{primary_score}。**",
                  "该分数表示严格检查下满足全部指令的提示词比例；其余三项仅供参考，不参与评分。", "",
                  "| 原生指标 | 原生分数（0–1） | 原生计数 |", "|---|---:|---:|"]
        for name in IF_METRICS:
            metric = accuracy.get("metrics", {}).get(name, {})
            lines.append(f"| {name} | {metric.get('score', '—')} | {metric.get('num', '—')} |")
        lines += ["", "指令级计数按指令聚合，不等同于提示词样本数；分数直接取 EvalScope 原生报告。"]
    else:
        lines.append("本次未执行精度阶段。")
    lines += ["", "## 异常与证据", ""]
    issues = summary.get("issues", []) + [f"并发 {r['concurrency']}：{issue}" for r in perf for issue in r.get("issues", [])]
    if accuracy:
        issues += ["精度：" + issue for issue in accuracy.get("issues", [])]
    lines += ["- " + md_cell(issue) for issue in issues] or ["- 未发现完整性异常。"]
    lines += ["", "- [结构化汇总](summary.json)", "- [环境与依赖版本](environment.json)",
              "- [评测输入配置](configuration.json)", "- 性能原生结果：`performance/c{1,2,4,8}/`；完整输入及其摘要：`performance/dataset.*`。",
              "- 精度原生结果：`accuracy/native/`；输入配置及解析后的默认配置：`accuracy/task_config.json`、`accuracy/resolved_config.json`。",
              "", "工具固定为 AIPerf 0.13.0、EvalScope 1.13.0。结果仅适用于本次模型、API 路径、样本和网络环境。", ""]
    (root / "report.md").write_text(redact("\n".join(lines), secret), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Infras 远程 API 性能与 IFBench 评测；缺少依赖须确认安装。")
    parser.add_argument("--mode", choices=["all", "perf", "accuracy", "check"], default="all")
    parser.add_argument("--base-url", type=normalize_api_url)
    parser.add_argument("--model")
    parser.add_argument("--tokenizer", default="builtin", help="模型 tokenizer 路径/HF 标识；默认 builtin（tiktoken/o200k_base）")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--env-dir", type=Path, default=Path.home() / ".cache/ai-infra-eval/envs")
    parser.add_argument("--perf-requests", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=2, help="EvalScope 评测 batch size（API 请求并发），默认 2")
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.env_dir = args.env_dir.expanduser().resolve()
    if args.output_dir:
        args.output_dir = args.output_dir.expanduser().resolve()
    if args.perf_requests < 1 or not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("请求数和超时必须为正数")
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size 必须为正整数")
    if args.mode != "check":
        if not args.base_url or not args.model:
            parser.error("评测和 dry-run 需要 --base-url 和 --model")
        if args.tokenizer == "builtin" and args.tokenizer_revision:
            parser.error("内置 tiktoken 不使用 --tokenizer-revision；请同时指定模型 --tokenizer")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stages = ["perf", "accuracy"] if args.mode in {"all", "check"} else [args.mode]
    if args.dry_run or args.mode == "check":
        status = dependency_status(args, stages)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        if args.dry_run:
            placeholder = Path("<OUTPUT>")
            if "perf" in stages:
                for concurrency in CONCURRENCIES:
                    print(shlex.join(aiperf_command(args, concurrency, placeholder / "performance/dataset.jsonl",
                                                   placeholder / f"performance/c{concurrency}", f"<RUN_ID>:c{concurrency}", "<API_KEY_FROM_ENV>")))
            if "accuracy" in stages:
                print(json.dumps(accuracy_config(args, placeholder / "accuracy"), ensure_ascii=False, indent=2))
            return 0
        return 0 if status["uv"] and all(v["ready"] for v in status["tools"].values()) else 2
    key = os.environ.get(args.api_key_env, "EMPTY")
    if not key or "\n" in key or "\r" in key:
        raise EvaluationError("API key 环境变量为空或包含换行。无鉴权时请取消设置该变量。")
    if args.output_dir and args.output_dir.exists():
        raise EvaluationError("输出目录已存在，请使用新的 --output-dir。")
    status = ensure_dependencies(args, stages)
    root = args.output_dir or Path("outputs") / dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.chmod(root, 0o700)
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration.update(isl=ISL, osl=OSL, concurrencies=list(CONCURRENCIES), warmup_requests=WARMUP,
                         accuracy_limit=LIMIT, accuracy_generation_config={"max_tokens": MAX_TOKENS})
    write_json(root / "configuration.json", configuration)
    write_json(root / "environment.json", {"platform": platform.platform(), "dependencies": status, "created_at": utc_now()})
    summary = {"started_at": utc_now(), "configuration": configuration, "status": "运行中", "issues": []}
    interrupted = False
    try:
        try:
            model_info = check_model(args.base_url, args.model, key)
            write_json(root / "model_info.json", model_info, key)
        except EvaluationError as error:
            summary["issues"].append(str(error))
            model_info = None
        if model_info is not None:
            if "perf" in stages:
                try:
                    summary["performance"] = run_performance(args, root, key, uuid.uuid4().hex)
                except (EvaluationError, OSError, ValueError) as error:
                    summary["issues"].append("性能阶段：" + str(error))
            if "accuracy" in stages:
                try:
                    summary["accuracy"] = run_accuracy(args, root, key, model_info)
                except (EvaluationError, OSError, ValueError) as error:
                    summary["accuracy"] = {"status": "异常", "issues": [str(error)], "artifacts": str(root / "accuracy")}
    except KeyboardInterrupt:
        interrupted = True
        summary["issues"].append("用户中断；已终止评测子进程，保留已完成及部分原生结果。")
    except Exception as error:
        summary["issues"].append(f"执行错误（{type(error).__name__}）：{error}")
    finally:
        checkpoint = root / "performance/summary.json"
        if "performance" not in summary and checkpoint.is_file():
            summary["performance"] = json.loads(checkpoint.read_text(encoding="utf-8"))
        summary["ended_at"] = utc_now()
        bad = summary["issues"] or any(r["status"] != "完整" for r in summary.get("performance", []))
        bad = bad or summary.get("accuracy", {}).get("status", "完整") != "完整"
        summary["status"] = "已中断" if interrupted else ("异常" if bad else "完整")
        sanitize_files(root, key)
        write_json(root / "summary.json", summary, key)
        render_report(root, summary, key)
        print(f"\n评测状态：{summary['status']}\nMarkdown 报告：{root / 'report.md'}", flush=True)
    return 130 if interrupted else (1 if summary["status"] != "完整" else 0)


if __name__ == "__main__":
    # These workers run inside the corresponding isolated tool environment.
    if len(sys.argv) > 1 and sys.argv[1] == "_dataset":
        dataset_worker(sys.argv[2], sys.argv[3] or None, Path(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    elif len(sys.argv) > 1 and sys.argv[1] == "_accuracy":
        accuracy_worker(Path(sys.argv[2]))
    else:
        def interrupted_signal(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted_signal)
        try:
            raise SystemExit(main())
        except (EvaluationError, OSError, ValueError) as error:
            print(f"错误：{redact(str(error), os.environ.get('OPENAI_API_KEY', ''))}", file=sys.stderr)
            raise SystemExit(2)
