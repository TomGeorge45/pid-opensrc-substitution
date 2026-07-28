"""Local-Qwen3-VL Anthropic-Messages-shaped client, used to drive the REAL
stage_13_run/stage_12_run drivers via `vlm_runner=` injection — the local-model
counterpart to `real_openai_client.py`'s `RealOpenAIMessagesClient`/`RealOpenAIRunner`,
targeting a Qwen3-VL-8B model + LoRA adapter running IN-PROCESS on the Colab GPU instead
of a live API call.

Both drivers call `runner._get_messages_client()` then talk to it via the raw
`client.messages.create(model=, max_tokens=, system=, tools=[...], tool_choice=...,
messages=[...])` Anthropic-Messages shape (same facts `real_openai_client.py` already
cites from entity_validation/driver.py + relation_validation/relation_validator.py):
  - `system` is a content-block list with `cache_control` (Anthropic-only, dropped here,
    same as the OpenAI translator).
  - `messages` is `[{"role": "user", "content": [...]}]`, content mixing an arbitrary
    number of `{"type": "image", ...}` blocks (stage 13: 1 image; stage 12: 3 images) and
    `{"type": "text", ...}` blocks — handled generically over count, same requirement
    `real_openai_client.py` already solved for the OpenAI side.

Qwen3-VL has no native tool-calling, so — same JSON-in-prompt technique this project's
Qwen-only Stage 4 notebook (`ArmL_QwenVL_FullStack_GPUOnly.ipynb`, section 6) already
uses — the tool schema is rendered into an explicit "respond with one fenced ```json
block" instruction, built directly from `tools[0]["input_schema"]` so the shape asked for
is schema-driven, not hand-typed per stage (this module is used for BOTH stage 13's
entity-verdict schema and stage 12's relation-verdict schema without any stage-specific
branching in the instruction-builder itself).

**Important caveat, discovered while building this (see notebook markdown for detail):**
`parse_json_common.parse_entity_verdict_json`/`parse_relation_verdict_json`'s own
docstrings say the v3-stage13/v3-relation adapters were trained on plain "keep"/"remove"
and "yes"/"no" text answers, NOT JSON. This client's PRIMARY path still asks for fenced
JSON (matching the task's specified technique and the driver's real declared schema), but
falls back to that same plain-text convention — reusing `parse_entity_verdict_json`/
`parse_relation_verdict_json` verbatim — if fenced-JSON extraction fails or the parsed
object doesn't carry the tool's key field. Whether the fenced-JSON path or the plain-text
fallback actually fires more often under the real adapters is UNTESTED — flagged, not
resolved, in this pass.
"""
from __future__ import annotations

import base64
import dataclasses
import json
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from ..backends.parse_json_common import _extract_json_text, parse_entity_verdict_json, parse_relation_verdict_json


def _system_to_text(system) -> str:
    """Same logic as real_openai_client._system_to_text — duplicated rather than imported
    so this module has no import-time coupling to the OpenAI-specific file."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for block in system:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _anthropic_content_to_qwen_content(content: list) -> list:
    """Translate a list of Anthropic content blocks (image/text) into Qwen chat-template
    content blocks (`{"type": "image", "image": PIL.Image}` / `{"type": "text", "text": ...}`).
    Handles an arbitrary number of image blocks generically (stage 13 sends 1, stage 12
    sends 3) — same requirement real_openai_client.py's `_anthropic_content_to_openai_input`
    already solved for the OpenAI side."""
    out = []
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "image":
            source = block.get("source") if isinstance(block, dict) else getattr(block, "source", None)
            data = (source or {}).get("data")
            img = Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
            out.append({"type": "image", "image": img})
        elif block_type == "text":
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            out.append({"type": "text", "text": text or ""})
        else:
            raise ValueError(f"LocalQwenMessagesClient: unhandled Anthropic content block type {block_type!r}")
    return out


def _anthropic_messages_to_qwen_content(messages: list) -> list:
    """Flatten the Anthropic `messages` list into ONE Qwen user-turn content list — both
    drivers send a single user turn, so there's nothing to preserve role-wise; only the
    content blocks (arbitrary count of image/text) matter."""
    out = []
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", [])
        if isinstance(content, str):
            out.append({"type": "text", "text": content})
        else:
            out.extend(_anthropic_content_to_qwen_content(content))
    return out


def build_json_instruction(tool_schema: dict) -> str:
    """Render an explicit fenced-```json instruction directly from `tool_schema["input_schema"]`
    (a JSON-Schema dict) — schema-driven, not hand-typed per stage, so the SAME renderer
    serves stage 13's entity-verdict schema and stage 12's relation-verdict schema. Mirrors
    the technique `ArmL_QwenVL_FullStack_GPUOnly.ipynb` section 6 already uses for stage 4's
    `detect_symbols` schema, generalized from that notebook's hand-written field list to an
    arbitrary schema's `properties`/`required`."""
    schema = tool_schema.get("input_schema", {}) or {}
    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []

    field_lines = []
    for key, spec in props.items():
        typ = spec.get("type", "any") if isinstance(spec, dict) else "any"
        desc = spec.get("description", "") if isinstance(spec, dict) else ""
        enum = spec.get("enum") if isinstance(spec, dict) else None
        typ_text = f"one of {json.dumps(enum)}" if enum else typ
        suffix = f"  // {desc}" if desc else ""
        field_lines.append(f'  "{key}": <{typ_text}>{suffix}')
    fields_block = ",\n".join(field_lines)

    return (
        f'You have NO tool-calling ability. Instead, respond with EXACTLY ONE fenced code '
        f'block, opened with ```json and closed with ```, containing nothing else outside '
        f'the fence, and nothing inside the fence except one JSON object matching this shape '
        f'(mirrors the "{tool_schema.get("name")}" tool schema):\n\n'
        f'{{\n{fields_block}\n}}\n\n'
        f'Required fields: {json.dumps(required)}. Do not add any other text, before or after '
        f'the fenced block.'
    )


class FakeResponse(SimpleNamespace):
    """Same duck-type as fake_llm.FakeResponse / real_openai_client.FakeResponse."""
    pass


def _fallback_payload(tool_schema: dict, raw_text: str) -> dict:
    """If fenced-JSON extraction fails (or the parsed object is missing the tool's key
    field), fall back to this project's own plain-text convention — reusing
    `parse_entity_verdict_json`/`parse_relation_verdict_json` verbatim, since their
    docstrings record that the v3-stage13/v3-relation adapters were actually TRAINED on
    plain "keep"/"remove" and "yes"/"no" answers, not JSON (see module docstring). Dispatches
    on which key the tool's declared schema carries — "keep" -> entity-verdict tool,
    "verdict" -> relation-verdict tool — rather than hardcoding tool names, so this stays
    schema-driven like `build_json_instruction`. Any other schema shape gets an empty dict
    (same "empty payload on failure" convention `RealOpenAIMessagesClient` already uses)."""
    props = (tool_schema.get("input_schema", {}) or {}).get("properties", {}) or {}
    if "keep" in props:
        outcome = parse_entity_verdict_json(raw_text, temp_id="_unused")
        if outcome.parse_failed or outcome.value is None:
            return {}
        d = dataclasses.asdict(outcome.value)
        d.pop("temp_id", None)
        return d
    if "verdict" in props:
        outcome = parse_relation_verdict_json(raw_text, relation_id="_unused")
        if outcome.parse_failed or outcome.value is None:
            return {}
        d = dataclasses.asdict(outcome.value)
        d.pop("relation_id", None)
        return d
    return {}


class LocalQwenMessagesClient:
    """Drop-in replacement for `fake_llm.FakeMessagesClient` / `real_openai_client.
    RealOpenAIMessagesClient` that calls a LOCAL Qwen3-VL-8B model (+ an active LoRA
    adapter, switched via `peft`'s multi-adapter `model.set_adapter(name)`) instead of a
    live API. One instance is bound to ONE active adapter name at construction time —
    build two instances (or call `.set_active_adapter(...)` before each stage's calls) to
    switch between the v3-stage13 and v3-relation adapters on the same base model.

    `generate_fn(qwen_content: list, max_new_tokens: int) -> str` is injected rather than
    hardcoding `model.generate(...)` here, so this module has no import-time torch
    dependency (mirrors `real_openai_client.py` only needing `openai` at import time) and
    so the SAME client class works whether the caller wraps `apply_chat_template` +
    `generate` directly or adds its own retry/timeout wrapper.

    Concurrency note (stage 12 dispatches its calls via `asyncio.gather`, per
    `fake_llm.py`'s docstring): `create()` below awaits nothing internally — it runs the
    (synchronous, GPU-bound) `generate_fn` call to completion before returning control to
    the event loop — so concurrent `asyncio.gather` callers naturally SERIALIZE on this one
    GPU model instance rather than truly overlapping. That's the safe behavior for a single
    CUDA context / KV cache, not an oversight."""

    def __init__(self, generate_fn, *, adapter_name: str | None = None, max_new_tokens: int = 1024):
        self._generate_fn = generate_fn
        self._adapter_name = adapter_name
        self._max_new_tokens = max_new_tokens
        self.calls = []  # recorded for debugging, mirrors FakeMessagesClient

    def set_active_adapter(self, adapter_name: str) -> None:
        self._adapter_name = adapter_name

    @property
    def messages(self):
        return _LocalQwenMessagesNamespace(self)


class _LocalQwenMessagesNamespace:
    def __init__(self, outer: LocalQwenMessagesClient):
        self._outer = outer

    async def create(self, *, model=None, max_tokens=None, system=None, tools=None,
                      tool_choice=None, messages=None, **_ignored):
        self._outer.calls.append({
            "model": model, "max_tokens": max_tokens, "system": system,
            "tools": tools, "tool_choice": tool_choice, "messages": messages,
            "adapter_name": self._outer._adapter_name,
        })

        if not tools:
            raise ValueError("LocalQwenMessagesClient requires exactly one tool (tool-forced call)")
        tool_schema = tools[0]
        tool_name = tool_schema["name"]

        system_text = _system_to_text(system)
        json_instruction = build_json_instruction(tool_schema)
        qwen_body_content = _anthropic_messages_to_qwen_content(messages or [])

        # Prepend system prompt + the schema-driven JSON instruction as one leading text
        # block; images/text from the driver's own user turn follow it in original order —
        # this is the same "text prefix + real content blocks" pattern the Qwen-only Stage 4
        # notebook already uses (system_prompt + "\n\n---\n\n" + user_text).
        prefix_text = system_text + "\n\n---\n\n" + json_instruction + "\n\n---\n\n"
        qwen_content = [{"type": "text", "text": prefix_text}] + qwen_body_content

        n_tokens = max_tokens or self._outer._max_new_tokens
        raw_text = self._outer._generate_fn(qwen_content, n_tokens)

        candidate = _extract_json_text(raw_text)
        payload = None
        if candidate is not None:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                payload = None

        props = (tool_schema.get("input_schema", {}) or {}).get("properties", {}) or {}
        key_field = "keep" if "keep" in props else ("verdict" if "verdict" in props else None)
        if not isinstance(payload, dict) or (key_field is not None and key_field not in payload):
            payload = _fallback_payload(tool_schema, raw_text)
        if payload is None:
            payload = {}

        usage = {"input_tokens": 0, "output_tokens": 0,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        return FakeResponse(
            content=[{"type": "tool_use", "name": tool_name, "input": payload}],
            usage=usage,
        )


class LocalQwenRunner:
    """Drop-in replacement for `fake_llm.FakeRunner` / `real_openai_client.RealOpenAIRunner`
    — `_get_messages_client()` returns a `LocalQwenMessagesClient` bound to whichever
    adapter was active when the runner was built (or last switched via
    `client.set_active_adapter(...)`)."""

    def __init__(self, client: LocalQwenMessagesClient):
        self._client = client

    def _get_messages_client(self):
        return self._client
