"""
Async vLLM client with concurrency control and throughput monitoring.

Handles both text-only and vision (image) requests via OpenAI-compatible API.
Automatically manages concurrency to maximize throughput without OOM.

Usage:
    client = VLLMClient("http://10.0.0.1:8000/v1", max_concurrent=40)
    results = await client.batch_chat(requests)
    client.print_stats()
"""

import asyncio
import base64
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


def encode_image_base64(image_path: str) -> str:
    """Encode a local image file to base64 data URI."""
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
    """Build OpenAI-format content list with text and optional images."""
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
        timeout: float = 300.0,
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

            self._client = AsyncOpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    async def _call_one(
        self,
        messages: List[Dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        request_id: str = "",
    ) -> Optional[str]:
        """Make a single API call with semaphore-controlled concurrency."""
        async with self.semaphore:
            client = await self._get_client()
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                result = response.choices[0].message.content
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
                self.stats.failed += 1
                self.stats.errors.append(f"{request_id}: {exc}")
                logger.warning("Request %s failed: %s", request_id, exc)
                return None

    async def batch_chat(
        self,
        requests: List[Dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> List[Optional[str]]:
        """Send a batch of requests with automatic concurrency control.

        Each request dict has:
            - "messages": list of message dicts
            - "id": optional request identifier
            - "max_tokens": optional per-request override
            - "temperature": optional per-request override
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
