"""Real-GPT-5.5 Anthropic-Messages-shaped client, used to drive the REAL
stage_13_run/stage_12_run drivers via `vlm_runner=` injection, exactly like
`fake_llm.py`'s FakeMessagesClient/FakeRunner but backed by a live OpenAI
Responses API call instead of a canned answer.

Both drivers call `runner._get_messages_client()` then talk to it via the raw
`client.messages.create(model=, max_tokens=, system=, tools=[...],
tool_choice=..., messages=[...])` Anthropic-Messages shape. This module
translates that shape into OpenAI's Responses API (`client.responses.create`)
and translates the reply back into the Anthropic-shaped `FakeResponse` the
drivers' defensive, getattr-with-dict-fallback parsers
(`_extract_tool_payload` / `_extract_tool_use`) already know how to read.

Confirmed by reading the real driver/prompt-builder source (entity_validation/
driver.py, relation_validation/relation_validator.py):
  - `system` is a content-block list: `[{"type": "text", "text": ..., "cache_control": {...}}]`
    (cache_control is Anthropic-only and has no OpenAI analogue - dropped here).
  - `messages` is always `[{"role": "user", "content": [...]}]` where content
    mixes `{"type": "image", "source": {"type": "base64", "media_type": ...,
    "data": ...}}` blocks (stage 13: 1 image; stage 12: 3 images) and
    `{"type": "text", "text": ...}` blocks. Order/count is arbitrary - the
    translator below handles any number of images generically.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from openai import AsyncOpenAI


def anthropic_tool_to_openai(tool_schema: dict) -> dict:
    """Same wire-format translation as poc_run_arm_p_v2.anthropic_tool_to_openai
    (Responses-API flat function-tool shape) - duplicated here rather than
    imported so this module has no import-time dependency on the v2 script
    (which builds a module-level OpenAI client requiring OPENAI_API_KEY at
    import time)."""
    return {
        "type": "function",
        "name": tool_schema["name"],
        "description": tool_schema.get("description", ""),
        "parameters": tool_schema["input_schema"],
        "strict": False,
    }


def _system_to_text(system) -> str:
    """Anthropic `system` is either a plain string or a content-block list.
    Concatenate all text blocks; drop cache_control (no OpenAI analogue)."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for block in system:
        if isinstance(block, dict):
            text = block.get("text")
        else:
            text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _anthropic_content_to_openai_input(content: list) -> list:
    """Translate a list of Anthropic content blocks (image/text) into OpenAI
    Responses-API input blocks. Handles an arbitrary number of image blocks
    (stage 13 sends 1, stage 12 sends 3) generically."""
    out = []
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "image":
            source = block.get("source") if isinstance(block, dict) else getattr(block, "source", None)
            media_type = (source or {}).get("media_type", "image/png")
            data = (source or {}).get("data")
            out.append({
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{data}",
            })
        elif block_type == "text":
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            out.append({"type": "input_text", "text": text or ""})
        else:
            raise ValueError(f"RealOpenAIMessagesClient: unhandled Anthropic content block type {block_type!r}")
    return out


def _anthropic_messages_to_openai_input(messages: list) -> list:
    """Translate the Anthropic `messages` list (role/content pairs) into
    Responses-API `input` items. Only `user` messages are expected from these
    drivers (both call sites send a single user turn), but this is written
    generically over role."""
    out = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "user")
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", [])
        if isinstance(content, str):
            out.append({"role": role, "content": [{"type": "input_text", "text": content}]})
        else:
            out.append({"role": role, "content": _anthropic_content_to_openai_input(content)})
    return out


class FakeResponse(SimpleNamespace):
    """Same duck-type as fake_llm.FakeResponse - enough of an anthropic.types.Message
    for _extract_tool_payload / _extract_tool_use and the usage helpers to work."""
    pass


class RealOpenAIMessagesClient:
    """Drop-in replacement for `fake_llm.FakeMessagesClient` that calls the REAL
    GPT-5.5 Responses API instead of a canned answer function. Same `.messages`
    property / `.messages.create(**kwargs)` shape."""

    def __init__(self, client: "AsyncOpenAI | None" = None, *, model: str = "gpt-5.5",
                 reasoning_effort: str = "low"):
        self._client = client or AsyncOpenAI()
        self._model = model
        self._reasoning_effort = reasoning_effort
        self.calls = []  # recorded for debugging, mirrors FakeMessagesClient

    @property
    def messages(self):
        return _RealMessagesNamespace(self)


class _RealMessagesNamespace:
    def __init__(self, outer: RealOpenAIMessagesClient):
        self._outer = outer

    async def create(self, *, model=None, max_tokens=1024, system=None, tools=None,
                      tool_choice=None, messages=None, **_ignored):
        self._outer.calls.append({
            "model": model, "max_tokens": max_tokens, "system": system,
            "tools": tools, "tool_choice": tool_choice, "messages": messages,
        })

        if not tools:
            raise ValueError("RealOpenAIMessagesClient requires exactly one tool (tool-forced call)")
        openai_tool = anthropic_tool_to_openai(tools[0])
        tool_name = openai_tool["name"]

        system_text = _system_to_text(system)
        openai_messages = _anthropic_messages_to_openai_input(messages or [])
        openai_input = [{"role": "system", "content": system_text}] + openai_messages

        oa_tool_choice = {"type": "function", "name": tool_name}
        if isinstance(tool_choice, dict) and tool_choice.get("name"):
            oa_tool_choice = {"type": "function", "name": tool_choice["name"]}

        resp = await self._outer._client.responses.create(
            model=self._outer._model,
            reasoning={"effort": self._outer._reasoning_effort},
            max_output_tokens=max_tokens,
            tools=[openai_tool],
            tool_choice=oa_tool_choice,
            input=openai_input,
        )

        call = next((item for item in resp.output if item.type == "function_call"), None)
        if call is None:
            # No tool call surfaced (e.g. refusal, incomplete) - return an empty
            # payload matching the tool's expected keys; drivers treat a missing/
            # unparseable payload defensively (see _extract_tool_payload /
            # _extract_tool_use), so an empty dict is safe here too.
            payload = {}
        else:
            try:
                payload = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError):
                payload = {}

        usage = getattr(resp, "usage", None)
        usage_dict = {
            "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

        return FakeResponse(
            content=[{"type": "tool_use", "name": tool_name, "input": payload}],
            usage=usage_dict,
        )


class RealOpenAIRunner:
    """Drop-in replacement for `fake_llm.FakeRunner` - `_get_messages_client()`
    returns a `RealOpenAIMessagesClient` instead of the fake one."""

    def __init__(self, client: RealOpenAIMessagesClient | None = None, **client_kwargs):
        self._client = client or RealOpenAIMessagesClient(**client_kwargs)

    def _get_messages_client(self):
        return self._client
