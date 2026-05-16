#!/usr/bin/env python3
"""Adaptive vLLM keepalive/load generator.

This is intentionally a real chat-completions load, not a /v1/models ping.
It keeps idle vLLM nodes awake and doing decode work, while backing off when
other jobs are already using the endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


DEFAULT_TARGETS = (
    "name=4c-122b,api_base=http://10.16.18.9:8000/v1,"
    "model=/home/tione/notebook/gaozhenkun/model/Qwen3.5-122B-A10B-FP8,"
    "min_active=192,max_own=256,idle_below=64,max_tokens=128,min_tokens=96,"
    "burst=48,max_waiting=256,max_kv=0.82",
    "name=8c-397b-175,api_base=http://10.16.12.175:8000/v1,"
    "model=/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8,"
    "min_active=256,max_own=320,idle_below=24,max_tokens=128,min_tokens=96,"
    "burst=48,max_waiting=256,max_kv=0.82",
    "name=8c-397b-172,api_base=http://10.16.10.172:8000/v1,"
    "model=/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8,"
    "min_active=256,max_own=320,idle_below=24,max_tokens=128,min_tokens=96,"
    "burst=48,max_waiting=256,max_kv=0.82",
    "name=8c-397b-160,api_base=http://10.16.11.160:8000/v1,"
    "model=/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8,"
    "min_active=256,max_own=320,idle_below=24,max_tokens=128,min_tokens=96,"
    "burst=48,max_waiting=256,max_kv=0.82",
)


@dataclass
class Metrics:
    ok: bool = False
    running: int = 0
    waiting: int = 0
    kv_cache: float = 0.0
    preemptions: float = 0.0
    prompt_tokens: float = 0.0
    generation_tokens: float = 0.0
    error: str = ""

    @property
    def active(self) -> int:
        return self.running + self.waiting


@dataclass
class Target:
    name: str
    api_base: str
    model: str = ""
    min_active: int = 128
    max_own: int = 160
    idle_below: int = 32
    max_tokens: int = 96
    min_tokens: int = 64
    burst: int = 32
    max_waiting: int = 128
    max_kv: float = 0.82
    request_timeout: float = 1800.0

    @property
    def metrics_url(self) -> str:
        base = self.api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return base + "/metrics"

    @property
    def chat_url(self) -> str:
        return self.api_base.rstrip("/") + "/chat/completions"

    @property
    def models_url(self) -> str:
        return self.api_base.rstrip("/") + "/models"


@dataclass
class TargetState:
    client: Any
    target: Target
    inflight: set[asyncio.Task] = field(default_factory=set)
    completed: int = 0
    failed: int = 0
    spawned: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_log: float = 0.0
    seq: int = 0
    simple_sampling: bool = False


def parse_target(raw: str) -> Target:
    values: dict[str, str] = {}
    for item in raw.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"bad target item {item!r}; expected key=value")
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()
    if "name" not in values or "api_base" not in values:
        raise ValueError(f"target needs name and api_base: {raw!r}")

    int_fields = {
        "min_active",
        "max_own",
        "idle_below",
        "max_tokens",
        "min_tokens",
        "burst",
        "max_waiting",
    }
    kwargs: dict[str, Any] = {}
    for field_name in Target.__dataclass_fields__:
        if field_name not in values:
            continue
        if field_name in int_fields:
            kwargs[field_name] = int(values[field_name])
        elif field_name in {"max_kv", "request_timeout"}:
            kwargs[field_name] = float(values[field_name])
        else:
            kwargs[field_name] = values[field_name]
    return Target(**kwargs)


def metric_float(text: str, name: str) -> float | None:
    prefix_label = name + "{"
    prefix_plain = name + " "
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(prefix_label) or line.startswith(prefix_plain):
            try:
                return float(line.split()[-1])
            except (IndexError, ValueError):
                return None
    return None


def parse_metrics(text: str) -> Metrics:
    running = metric_float(text, "vllm:num_requests_running")
    waiting = metric_float(text, "vllm:num_requests_waiting")
    kv_cache = metric_float(text, "vllm:kv_cache_usage_perc")
    preemptions = metric_float(text, "vllm:num_preemptions_total")
    prompt_tokens = metric_float(text, "vllm:prompt_tokens_total")
    generation_tokens = metric_float(text, "vllm:generation_tokens_total")
    return Metrics(
        ok=True,
        running=int(running or 0),
        waiting=int(waiting or 0),
        kv_cache=float(kv_cache or 0.0),
        preemptions=float(preemptions or 0.0),
        prompt_tokens=float(prompt_tokens or 0.0),
        generation_tokens=float(generation_tokens or 0.0),
    )


async def fetch_metrics(state: TargetState) -> Metrics:
    try:
        resp = await state.client.get(state.target.metrics_url, timeout=10.0)
        resp.raise_for_status()
        return parse_metrics(resp.text)
    except Exception as exc:  # noqa: BLE001 - keepalive should keep running
        return Metrics(error=str(exc))


async def ensure_model(state: TargetState) -> None:
    if state.target.model:
        return
    resp = await state.client.get(state.target.models_url, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"{state.target.name}: /v1/models returned no models")
    state.target.model = str(models[0]["id"])


def build_messages(target: Target, seq: int) -> list[dict[str, str]]:
    seed = random.getrandbits(64)
    prefix = " ".join(str(random.randint(10000, 99999)) for _ in range(160))
    return [
        {
            "role": "system",
            "content": (
                "You are a vLLM load-generation worker. Return only a long "
                "comma-separated integer sequence. Do not explain."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Keepalive target={target.name} request={seq} seed={seed}. "
                f"Continue this sequence for at least {target.max_tokens * 2} "
                f"integers and do not stop early: {prefix}"
            ),
        },
    ]


async def post_completion(state: TargetState, seq: int) -> tuple[int, int]:
    target = state.target
    body: dict[str, Any] = {
        "model": target.model,
        "messages": build_messages(target, seq),
        "max_tokens": target.max_tokens,
        "temperature": 0.8,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if not state.simple_sampling:
        body["min_tokens"] = min(target.min_tokens, target.max_tokens)
        body["ignore_eos"] = True

    resp = await state.client.post(
        target.chat_url,
        json=body,
        timeout=target.request_timeout,
    )
    if resp.status_code == 400 and not state.simple_sampling:
        state.simple_sampling = True
        body.pop("min_tokens", None)
        body.pop("ignore_eos", None)
        resp = await state.client.post(
            target.chat_url,
            json=body,
            timeout=target.request_timeout,
        )
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


async def reap_done(state: TargetState) -> None:
    done = {task for task in state.inflight if task.done()}
    if not done:
        return
    state.inflight.difference_update(done)
    for task in done:
        try:
            prompt_tokens, completion_tokens = await task
        except Exception:
            state.failed += 1
        else:
            state.completed += 1
            state.prompt_tokens += prompt_tokens
            state.completion_tokens += completion_tokens


def plan_spawn(state: TargetState, metrics: Metrics) -> tuple[int, str, int]:
    target = state.target
    own = len(state.inflight)
    external_active = max(0, metrics.active - own)
    if not metrics.ok:
        return 0, f"metrics_error={metrics.error[:120]}", external_active
    if external_active > target.idle_below:
        return 0, f"external_active>{target.idle_below}", external_active
    if metrics.waiting > target.max_waiting:
        return 0, f"waiting>{target.max_waiting}", external_active
    if metrics.kv_cache >= target.max_kv:
        return 0, f"kv>={target.max_kv:.2f}", external_active

    total_needed = max(0, target.min_active - metrics.active)
    own_headroom = max(0, target.max_own - own)
    spawn = min(total_needed, own_headroom, target.burst)
    return spawn, "spawn" if spawn else "at_target", external_active


async def target_loop(
    state: TargetState,
    stop: asyncio.Event,
    metrics_interval: float,
    log_interval: float,
    jsonl_path: str,
) -> None:
    await ensure_model(state)
    while not stop.is_set():
        await reap_done(state)
        metrics = await fetch_metrics(state)
        spawn_n, reason, external_active = plan_spawn(state, metrics)
        for _ in range(spawn_n):
            state.seq += 1
            task = asyncio.create_task(post_completion(state, state.seq))
            state.inflight.add(task)
            state.spawned += 1

        now = time.time()
        if now - state.last_log >= log_interval:
            row = {
                "ts": datetime.now().strftime("%F %T"),
                "target": state.target.name,
                "api_base": state.target.api_base,
                "running": metrics.running,
                "waiting": metrics.waiting,
                "active": metrics.active,
                "external_active_est": external_active,
                "own_inflight": len(state.inflight),
                "kv_cache": round(metrics.kv_cache, 4),
                "preemptions": metrics.preemptions,
                "spawned_total": state.spawned,
                "completed": state.completed,
                "failed": state.failed,
                "completion_tokens": state.completion_tokens,
                "spawn_now": spawn_n,
                "reason": reason,
            }
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if jsonl_path:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            state.last_log = now

        try:
            await asyncio.wait_for(stop.wait(), timeout=metrics_interval)
        except asyncio.TimeoutError:
            pass

    for task in state.inflight:
        task.cancel()
    await asyncio.gather(*state.inflight, return_exceptions=True)


async def amain(args: argparse.Namespace) -> None:
    import httpx

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    targets = [parse_target(raw) for raw in (args.target or DEFAULT_TARGETS)]
    states: list[TargetState] = []
    for target in targets:
        limits = httpx.Limits(
            max_connections=max(64, target.max_own + 32),
            max_keepalive_connections=max(64, target.max_own),
            keepalive_expiry=120.0,
        )
        client = httpx.AsyncClient(
            timeout=target.request_timeout,
            limits=limits,
            trust_env=False,
            headers={"Authorization": "Bearer placeholder"},
        )
        states.append(TargetState(client=client, target=target))

    try:
        tasks = [
            asyncio.create_task(
                target_loop(
                    state,
                    stop,
                    metrics_interval=args.metrics_interval,
                    log_interval=args.log_interval,
                    jsonl_path=args.jsonl,
                )
            )
            for state in states
        ]
        await asyncio.gather(*tasks)
    finally:
        for state in states:
            await state.client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="append",
        help=(
            "Comma-separated target config, e.g. "
            "name=4c,api_base=http://host:8000/v1,model=/path,"
            "min_active=192,max_own=256,idle_below=64"
        ),
    )
    parser.add_argument("--metrics-interval", type=float, default=3.0)
    parser.add_argument("--log-interval", type=float, default=15.0)
    parser.add_argument("--jsonl", default="")
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
