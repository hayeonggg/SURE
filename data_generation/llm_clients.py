"""Async LLM clients (OpenAI, Anthropic, xAI) with retry / rate-limit handling.

Used for GPT-4o candidate generation and the three LLM judges.
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import config
from utils import get_logger

log = get_logger("llm")


class _RetryExhausted(RuntimeError):
    pass


async def _with_retries(
    label: str,
    coro_factory,
    max_retries: int | None = None,
    initial_backoff: float | None = None,
) -> Any:
    """`coro_factory()` -> coroutine. Re-create coroutine on each retry."""
    max_retries = max_retries or config.API.max_retries
    backoff = initial_backoff or config.API.initial_backoff
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep_s = backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            log.warning("[%s] attempt %d/%d failed: %s — sleep %.1fs",
                        label, attempt, max_retries, e, sleep_s)
            await asyncio.sleep(sleep_s)
    raise _RetryExhausted(f"{label}: {last_err}")


# ---------------- OpenAI ----------------
class OpenAIClient:
    def __init__(self) -> None:
        from openai import AsyncOpenAI
        if not config.API.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = AsyncOpenAI(
            api_key=config.API.openai_api_key,
            timeout=config.API.request_timeout,
        )
        self._sem = asyncio.Semaphore(config.API.max_concurrency)

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        response_format_json: bool = False,
    ) -> str:
        async def _call():
            async with self._sem:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                }
                if response_format_json:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = await self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
        return await _with_retries(f"openai:{model}", _call)


# ---------------- Anthropic ----------------
class AnthropicClient:
    def __init__(self) -> None:
        from anthropic import AsyncAnthropic
        if not config.API.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = AsyncAnthropic(
            api_key=config.API.anthropic_api_key,
            timeout=config.API.request_timeout,
        )
        # Anthropic uses a separate (lower) concurrency limit
        anthropic_concurrency = (
            config.API.anthropic_max_concurrency or config.API.max_concurrency
        )
        self._sem = asyncio.Semaphore(anthropic_concurrency)
        log.info("AnthropicClient ready (concurrency=%d)", anthropic_concurrency)

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        async def _call():
            async with self._sem:
                msg = await self._client.messages.create(
                    model=model,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # return the first text block only
                for block in msg.content:
                    if getattr(block, "type", None) == "text":
                        return block.text
                return ""
        return await _with_retries(f"anthropic:{model}", _call)


# ---------------- xAI (Grok) ----------------
class XAIClient:
    """The xAI API is OpenAI-compatible (only base_url differs), so AsyncOpenAI is reused."""

    def __init__(self) -> None:
        from openai import AsyncOpenAI
        if not config.API.xai_api_key:
            raise RuntimeError(
                "XAI_API_KEY is not set.\n"
                "  export XAI_API_KEY=xai-...\n"
                "Get a key at https://console.x.ai/."
            )
        self._client = AsyncOpenAI(
            api_key=config.API.xai_api_key,
            base_url="https://api.x.ai/v1",
            timeout=config.API.request_timeout,
        )
        self._sem = asyncio.Semaphore(config.API.max_concurrency)
        log.info("XAIClient ready (concurrency=%d)", config.API.max_concurrency)

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        response_format_json: bool = False,
    ) -> str:
        async def _call():
            async with self._sem:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                }
                if response_format_json:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = await self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
        return await _with_retries(f"xai:{model}", _call)


# ---------------- Dispatcher ----------------
_CLIENTS: dict[str, Any] = {}


def get_client(provider: str):
    if provider not in _CLIENTS:
        if provider == "openai":
            _CLIENTS[provider] = OpenAIClient()
        elif provider == "anthropic":
            _CLIENTS[provider] = AnthropicClient()
        elif provider == "xai":
            _CLIENTS[provider] = XAIClient()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    return _CLIENTS[provider]


async def call_model(
    spec: "config.ModelSpec",
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """Provider-agnostic single completion call."""
    client = get_client(spec.provider)
    if spec.provider in ("openai", "xai"):
        # both use the OpenAI-compatible chat.completions API
        return await client.complete(
            spec.model_id, system, user,
            temperature=temperature, response_format_json=json_mode,
        )
    if spec.provider == "anthropic":
        return await client.complete(
            spec.model_id, system, user, temperature=temperature,
        )
    raise ValueError(spec.provider)


def parse_json_loose(text: str) -> dict | None:
    """Judges may wrap JSON in ```json blocks; salvage gracefully."""
    if not text:
        return None
    text = text.strip()
    # strip markdown fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    # find first { and last }
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        text = text[s : e + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
