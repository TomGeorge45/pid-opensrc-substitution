"""Real GPT-5.5-low `call_llm` adapter, matching the exact contract
`pnid_pipeline.hierarchy.apply_hierarchy` (and `build_qwen_call_llm`) already use:
`call_llm(prompt, schema_model, *, images=None, model=None, max_tokens=None,
temperature=0.0) -> BaseModel`.

Why this didn't already exist: `pnid_pipeline.llm_proxy.build_call_llm` (gap #12's
originally-assumed prod adapter) talks to an Anthropic-Messages-shaped `/v1/messages`
endpoint via a LiteLLM-style proxy (`MODEL_PROXY_URL`/`MODEL_PROXY_MASTER_KEY`) — that proxy
isn't running here, and only a raw `OPENAI_API_KEY` is available (confirmed via direct env
check). `real_openai_client.py` (used elsewhere in this repo) is a DIFFERENT adapter for a
DIFFERENT contract — Anthropic-Messages tool-use blocks, built for intelligence-agent's
stage_13/12 drivers, not extraction-agent's simple prompt+pydantic-schema contract. So this
is a small, new, direct OpenAI Responses-API client for hierarchy.py's specific shape.

Usage tracking matches the same `{model: {"in","cache_w","cache_r","out","calls"}}` shape
`llm_proxy.snapshot`/`delta`/`usage_cost` (gap #20) already expect, so cost accounting works
identically to the other two arms without any special-casing.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, List, Optional, Type

from openai import AsyncOpenAI
from pydantic import BaseModel


def _schema_instruction(schema_model: Type[BaseModel]) -> str:
    schema = schema_model.model_json_schema()
    return (f"\n\nRespond with a SINGLE JSON object that conforms to this JSON Schema. "
            f"No prose, no markdown fences — JSON only.\n{json.dumps(schema)}")


def _extract_json_text(text: str) -> Optional[str]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start:end + 1]


_REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _split_model_effort(model_name: str, default_effort: str) -> tuple:
    """"gpt-5.5-low" -> ("gpt-5.5", "low") — the real OpenAI Responses API takes the base
    model id and reasoning effort as SEPARATE fields (confirmed via direct read of
    `real_openai_client.py::RealOpenAIRunner`, which already drives real GPT-5.5-low calls
    elsewhere in this repo as `model="gpt-5.5", reasoning={"effort": "low"}` — "gpt-5.5-low"
    as a single string is a benchmark-internal shorthand, not a real OpenAI model id (a raw
    API call with that string 400s: "model 'gpt-5.5-low' does not exist", confirmed live)."""
    for suffix in _REASONING_EFFORTS:
        if model_name.endswith(f"-{suffix}"):
            return model_name[: -(len(suffix) + 1)], suffix
    return model_name, default_effort


def build_openai_call_llm(*, api_key: Optional[str] = None,
                          default_model: str = "gpt-5.5",
                          default_effort: str = "low") -> Callable[..., Any]:
    client = AsyncOpenAI(api_key=api_key)
    usage: dict = {}
    lock = asyncio.Lock()

    async def call_llm(
        prompt: str,
        schema_model: Type[BaseModel],
        *,
        images: Optional[List[str]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        **_ignored: Any,
    ) -> BaseModel:
        base_model, effort = _split_model_effort(model or default_model, default_effort)
        content: List[dict] = [{"type": "input_text",
                                "text": prompt + _schema_instruction(schema_model)}]
        for b64 in images or []:
            content.append({"type": "input_image",
                            "image_url": f"data:image/png;base64,{b64}"})

        async with lock:
            resp = await client.responses.create(
                model=base_model,
                input=[{"role": "user", "content": content}],
                max_output_tokens=max_tokens or 4096,
                reasoning={"effort": effort},
            )

        slot = usage.setdefault(base_model, {"in": 0, "cache_w": 0, "cache_r": 0,
                                              "out": 0, "calls": 0})
        u = getattr(resp, "usage", None)
        if u is not None:
            slot["in"] += int(getattr(u, "input_tokens", 0) or 0)
            slot["out"] += int(getattr(u, "output_tokens", 0) or 0)
            cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0)
            slot["cache_r"] += int(cached or 0)
        slot["calls"] += 1

        raw_text = getattr(resp, "output_text", "") or ""
        try:
            json_text = _extract_json_text(raw_text)
            if json_text is None:
                raise ValueError("no JSON-like content in GPT-5.5-low output")
            return schema_model.model_validate(json.loads(json_text))
        except Exception:
            # Same honest-failure convention as build_qwen_call_llm.
            return schema_model()

    call_llm.usage = usage  # type: ignore[attr-defined]
    return call_llm
