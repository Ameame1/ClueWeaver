"""OpenAI-compatible async client for local ClueWeaver endpoints."""
from __future__ import annotations
import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return None


@dataclass
class ModelSpec:
    name: str
    model_id: str
    base_url: str
    api_key: str = "EMPTY"
    max_input_tokens: int = 32000
    request_timeout: float = 180.0
    enable_thinking: bool | None = None


MODELS: dict[str, ModelSpec] = {
    "qwen3-4b": ModelSpec(
        name="qwen3-4b",
        model_id=os.environ.get("QWEN_MODEL_NAME", "qwen3-4b-instruct"),
        base_url=os.environ.get("QWEN_BASE", "http://localhost:8000/v1"),
        api_key=os.environ.get("QWEN_KEY", "EMPTY"),
        max_input_tokens=32000,
        request_timeout=180.0,
        enable_thinking=_env_bool("QWEN_ENABLE_THINKING"),
    ),
    "qwen3-4b-finder": ModelSpec(
        name="qwen3-4b-finder",
        model_id=os.environ.get("QWEN_FINDER_MODEL_NAME", "qwen3-4b-finder"),
        base_url=os.environ.get("QWEN_FINDER_BASE", "http://localhost:8001/v1"),
        api_key=os.environ.get("QWEN_FINDER_KEY", "EMPTY"),
        max_input_tokens=12000,
        request_timeout=180.0,
        enable_thinking=_env_bool("QWEN_FINDER_ENABLE_THINKING"),
    ),
}


class LLMClient:
    def __init__(self, spec: ModelSpec, max_concurrency: int = 8):
        self.spec = spec
        self._client = AsyncOpenAI(
            api_key=spec.api_key,
            base_url=spec.base_url,
            timeout=spec.request_timeout,
        )
        self._sem = asyncio.Semaphore(max_concurrency)

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> str:
        async with self._sem:
            for attempt in range(retries):
                try:
                    extra_body = {}
                    if self.spec.enable_thinking is not None:
                        extra_body["enable_thinking"] = self.spec.enable_thinking
                    kwargs = {}
                    if extra_body:
                        kwargs["extra_body"] = extra_body
                    resp = await self._client.chat.completions.create(
                        model=self.spec.model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    return resp.choices[0].message.content or ""
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
            return ""

    async def ask(self, prompt: str, system: Optional[str] = None, **kw) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return await self.chat(msgs, **kw)


def get_client(model_name: str = "qwen3-4b", max_concurrency: int = 8) -> LLMClient:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name}. Known: {list(MODELS)}")
    return LLMClient(MODELS[model_name], max_concurrency=max_concurrency)
