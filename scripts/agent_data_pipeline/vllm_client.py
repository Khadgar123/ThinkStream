"""
Async vLLM client with concurrency control and throughput monitoring.

Handles text-only, image, and pre-sampled-video requests via the
OpenAI-compatible API.
Automatically manages concurrency to maximize throughput without OOM.

Usage:
    client = VLLMClient("http://10.0.0.1:8000/v1", max_concurrent=40)
    results = await client.batch_chat(requests)
    client.print_stats()
"""

import asyncio
import base64
import functools
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestStats:
    total: int = 0
    completed: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        return (self.end_time or time.time()) - self.start_time

    @property
    def throughput_rps(self) -> float:
        return self.completed / max(self.elapsed, 0.001)

    @property
    def throughput_tps(self) -> float:
        return self.total_output_tokens / max(self.elapsed, 0.001)


# v12.12 (2026-05-01): LRU cache.
#
# Pass2's sliding visual_window keeps each frame in the prompt for 16
# consecutive obs requests (VISUAL_WINDOW_CHUNKS). Without caching, every
# frame is read + base64-encoded 16× per video. At 1024 concurrent videos
# the redundant disk I/O + CPU encode is the dominant non-GPU cost in the
# observation hot path.
#
# Cache key: absolute path string. Frames extracted to data/agent_v5/frames/
# are immutable for the duration of a run, so path equality ⇒ content
# equality. maxsize=16384 ≈ 16 hot frames × 1024 active videos peak;
# at avg 80KB base64 string ⇒ ~1.3GB RSS budget (acceptable on the
# server already running a multi-hundred-GB teacher).
_BASE64_CACHE_MAXSIZE = 16384


@functools.lru_cache(maxsize=_BASE64_CACHE_MAXSIZE)
def encode_image_base64(image_path: str) -> str:
    """Encode a local image file to base64 data URI (LRU-cached by path)."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    suffix = Path(image_path).suffix.lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(
        suffix.lstrip("."), "jpeg"
    )
    return f"data:image/{mime};base64,{data}"


def build_content_with_images(
    text: str, image_paths: Optional[List[str]] = None
) -> list:
    """Build OpenAI-format content list with text and optional images.

    Streaming pass2 does not use this helper: it sends a single
    {"type": "video"} block made from pre-extracted frame data URIs plus
    video_metadata. That path avoids server-side raw-video decoding and lets
    Qwen3-VL receive official fps/frames_indices timestamp anchors.
    """
    content = []
    if image_paths:
        for img_path in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": encode_image_base64(img_path)},
            })
    content.append({"type": "text", "text": text})
    return content


class VLLMClient:
    """Async client for vLLM with concurrency control."""

    def __init__(
        self,
        api_base: str,
        model: str = "",
        max_concurrent: int = 40,
        api_key: str = "placeholder",
        timeout: float = 5400.0,  # 90 min — safety net for thinking-enabled passes
                                    # (1A/1B/2/3A still use thinking for quality).
                                    # pass3c now disables thinking + caps max_tokens
                                    # so it won't approach this anyway.
    ):
        self.api_base = api_base
        self.model = model
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stats = RequestStats()
        self._client = None

    async def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            import httpx

            # v9.5: explicit httpx connection pool sized to the semaphore
            # cap so the SDK path doesn't deadlock when other passes also
            # cap concurrency >100 (default httpx max_connections).
            limits = httpx.Limits(
                max_connections=max(2048, self.max_concurrent * 2),
                max_keepalive_connections=max(512, self.max_concurrent),
                keepalive_expiry=120.0,
            )
            transport = httpx.AsyncHTTPTransport(limits=limits)
            sdk_httpx = httpx.AsyncClient(
                transport=transport, timeout=self.timeout,
            )
            self._client = AsyncOpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
                http_client=sdk_httpx,
            )
        return self._client

    async def _get_httpx_client(self):
        """httpx client used for raw-body POST when SDK extra_body merging
        doesn't reach vLLM (verified failure mode for chat_template_kwargs
        on the 397B server: SDK extra_body silently ignored, raw body
        with chat_template_kwargs at top level works).

        v9.5: connection-pool limits sized to allow concurrent=1024 on
        pass1a without CLOSE_WAIT deadlocks. httpx default is 100
        max_connections / 20 keepalive — far below our semaphore cap.
        """
        if not hasattr(self, "_httpx_client") or self._httpx_client is None:
            import httpx
            limits = httpx.Limits(
                max_connections=max(2048, self.max_concurrent * 2),
                max_keepalive_connections=max(512, self.max_concurrent),
                keepalive_expiry=120.0,
            )
            self._httpx_client = httpx.AsyncClient(
                base_url=self.api_base, timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"},
                limits=limits,
            )
        return self._httpx_client

    async def _call_one_raw(
        self, messages, max_tokens, temperature, request_id, enable_thinking,
        mm_processor_kwargs: Optional[Dict] = None,
    ):
        """Raw POST path — bypasses OpenAI SDK so chat_template_kwargs
        actually reaches vLLM. Returns (content_str_or_None, prompt_tokens,
        completion_tokens). Used when enable_thinking is non-None or when
        mm_processor_kwargs is supplied (vLLM 0.7+ accepts this at request
        body top level for per-request smart_resize bounds).
        """
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        if mm_processor_kwargs:
            # v12.12 (2026-05-02): per-request Qwen3-VL smart_resize bounds.
            # vLLM ≥ 0.7.3 forwards top-level mm_processor_kwargs to
            # Qwen3VLProcessingInfo._get_vision_info, which sets
            # size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
            # and smart_resize aspect-preserves the image into that band.
            # Putting these inside `image_url` is silently dropped — must
            # be at body top level (here) or in OpenAI SDK extra_body.
            body["mm_processor_kwargs"] = dict(mm_processor_kwargs)
        client = await self._get_httpx_client()
        resp = await client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        # vLLM without --reasoning-parser can put thinking in "reasoning" and
        # leave content null. Only use that fallback when the caller did not
        # explicitly disable thinking; otherwise we would inject hidden CoT into
        # pass outputs.
        if not content and enable_thinking is not False:
            content = msg.get("reasoning", "") or msg.get("reasoning_content", "")
        usage = data.get("usage") or {}
        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    async def _call_one(
        self,
        messages: List[Dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        request_id: str = "",
        max_retries: int = 3,
        enable_thinking: Optional[bool] = None,
        mm_processor_kwargs: Optional[Dict] = None,
    ) -> Optional[str]:
        """Make a single API call with semaphore-controlled concurrency and retry.

        Args:
            enable_thinking: when False, ask vLLM to skip the reasoning
              phase via Qwen3 chat-template kwargs (`<think></think>`
              empty placeholder injected by the chat template).
              v9.5: routes through raw httpx POST (not the OpenAI SDK)
              because the SDK's `extra_body` is silently dropped on the
              397B server — verified curl-vs-SDK A/B by user. SDK path
              kept for `enable_thinking is None` (no-op = server default).
            mm_processor_kwargs: v12.12 — per-request Qwen3-VL smart_resize
              bounds, e.g. {"min_pixels": 130_000, "max_pixels": 220_000}.
              When set, routes through raw POST (same reason as
              enable_thinking — SDK extra_body unreliable). vLLM forwards
              this to the Qwen3-VL processor for aspect-preserving resize.
        """
        # When the caller explicitly toggles thinking OR sets mm_processor_kwargs,
        # take the raw POST path so the extra fields land at request-body top level.
        if enable_thinking is not None or mm_processor_kwargs is not None:
            async with self.semaphore:
                for attempt in range(max_retries):
                    try:
                        result, ptok, ctok = await self._call_one_raw(
                            messages, max_tokens, temperature, request_id,
                            enable_thinking,
                            mm_processor_kwargs=mm_processor_kwargs,
                        )
                        self.stats.completed += 1
                        self.stats.total_input_tokens += ptok
                        self.stats.total_output_tokens += ctok
                        if self.stats.completed % 50 == 0:
                            logger.info(
                                "Progress: %d/%d completed (%.1f req/s, %.1f tok/s)",
                                self.stats.completed, self.stats.total,
                                self.stats.throughput_rps, self.stats.throughput_tps,
                            )
                        return result
                    except Exception as e:
                        logger.warning(
                            "raw-call attempt %d failed [%s]: %s",
                            attempt + 1, request_id, e,
                        )
                        if attempt >= max_retries - 1:
                            self.stats.failed += 1
                            self.stats.errors.append(f"{request_id}: {e}")
                            return None
                        await asyncio.sleep(min(2 ** attempt, 30))
                return None

        async with self.semaphore:
            client = await self._get_client()
            for attempt in range(max_retries):
                try:
                    create_kwargs = dict(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    response = await client.chat.completions.create(**create_kwargs)
                    msg = response.choices[0].message
                    result = getattr(msg, "content", None) or ""
                    # Fallback when vLLM reasoning-parser is absent
                    if not result:
                        result = getattr(msg, "reasoning", "") or ""
                    usage = response.usage
                    self.stats.completed += 1
                    if usage:
                        self.stats.total_input_tokens += usage.prompt_tokens
                        self.stats.total_output_tokens += usage.completion_tokens

                    # Progress log every 50 requests
                    if self.stats.completed % 50 == 0:
                        logger.info(
                            "Progress: %d/%d completed (%.1f req/s, %.1f tok/s)",
                            self.stats.completed,
                            self.stats.total,
                            self.stats.throughput_rps,
                            self.stats.throughput_tps,
                        )
                    return result
                except Exception as exc:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(
                            "Request %s attempt %d failed: %s, retrying in %ds",
                            request_id, attempt + 1, exc, wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        self.stats.failed += 1
                        self.stats.errors.append(f"{request_id}: {exc}")
                        logger.warning("Request %s failed after %d attempts: %s", request_id, max_retries, exc)
                        return None

    async def batch_chat(
        self,
        requests: List[Dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        enable_thinking: Optional[bool] = None,
        mm_processor_kwargs: Optional[Dict] = None,
    ) -> List[Optional[str]]:
        """Send a batch of requests with automatic concurrency control.

        Each request dict has:
            - "messages": list of message dicts
            - "id": optional request identifier
            - "max_tokens": optional per-request override
            - "temperature": optional per-request override
            - "enable_thinking": optional per-request override
            - "mm_processor_kwargs": optional per-request Qwen3-VL
              smart_resize bounds (v12.12)
        """
        self.stats = RequestStats(total=len(requests), start_time=time.time())
        logger.info(
            "Starting batch: %d requests, max_concurrent=%d",
            len(requests), self.max_concurrent,
        )

        tasks = []
        for i, req in enumerate(requests):
            task = self._call_one(
                messages=req["messages"],
                max_tokens=req.get("max_tokens", max_tokens),
                temperature=req.get("temperature", temperature),
                request_id=req.get("id", f"req_{i}"),
                enable_thinking=req.get("enable_thinking", enable_thinking),
                mm_processor_kwargs=req.get("mm_processor_kwargs", mm_processor_kwargs),
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        self.stats.end_time = time.time()
        return results

    def print_stats(self):
        s = self.stats
        print(f"\n{'='*60}")
        print(f"vLLM Batch Stats")
        print(f"{'='*60}")
        print(f"  Total requests:    {s.total}")
        print(f"  Completed:         {s.completed}")
        print(f"  Failed:            {s.failed}")
        print(f"  Elapsed:           {s.elapsed:.1f}s")
        print(f"  Throughput:        {s.throughput_rps:.2f} req/s")
        print(f"  Token throughput:  {s.throughput_tps:.1f} tok/s")
        print(f"  Input tokens:      {s.total_input_tokens:,}")
        print(f"  Output tokens:     {s.total_output_tokens:,}")
        if s.errors:
            print(f"  Errors ({len(s.errors)}):")
            for e in s.errors[:5]:
                print(f"    {e}")
        print(f"{'='*60}")


async def stress_test(
    api_base: str,
    model: str,
    num_requests: int = 10,
    max_concurrent: int = 4,
    prompt: str = "Say hello in one word.",
    max_tokens: int = 32,
) -> RequestStats:
    """Stress test the vLLM endpoint to find max safe concurrency.

    Usage:
        python -c "
        import asyncio
        from scripts.agent_data_pipeline.vllm_client import stress_test
        stats = asyncio.run(stress_test(
            'http://10.0.0.1:8000/v1', 'Qwen/Qwen3.5-397B-A17B-FP8',
            num_requests=20, max_concurrent=8
        ))
        "
    """
    client = VLLMClient(api_base, model, max_concurrent=max_concurrent)
    requests = [
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "id": f"stress_{i}",
        }
        for i in range(num_requests)
    ]
    await client.batch_chat(requests)
    client.print_stats()
    return client.stats
