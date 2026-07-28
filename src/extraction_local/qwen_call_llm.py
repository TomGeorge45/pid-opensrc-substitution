"""`call_llm`-shaped callable matching the EXACT contract used by
`pnid_pipeline/grounded_read.py` (read_shapes/read_regions) and
`pnid_pipeline/llm_proxy.py` (build_call_llm):

    await call_llm(prompt: str, schema_model: Type[BaseModel],
                    images: list[str_b64], model: str, max_tokens: int
                    ) -> schema_model_instance

plus a `.usage` dict attribute of shape
    {model_name: {"in": int, "cache_w": int, "cache_r": int, "out": int, "calls": int}}
(exact shape confirmed against llm_proxy.py's `_record`/`snapshot`/`delta`, see
Extraction_Agent_Local_Plan.md §11 build-log item 7).

`generate_fn(prompt_text: str, images: list[str]) -> str` is the injection point —
a FAKE for Phase 1 testing, a real Qwen3-VL `generate()` wrapper later (Phase 2, GPU).
Local generation is inherently serial, so concurrency (the pipeline awaits under
Semaphore(6)) collapses to 1 via an asyncio.Lock — documented, not silently hidden.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Callable, List, Optional, Type, get_args, get_origin

from pydantic import BaseModel

# Reuse the project's proven JSON-answer extractor rather than reinventing one.
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from e2e_bench.backends.parse_json_common import _extract_json_text  # noqa: E402

GenerateFn = Callable[[str, List[str]], str]


def _default_instance(schema_model: Type[BaseModel]) -> BaseModel:
    """Empty/default instance for the honest-failure path. `_OneTag`/`_ManyTags` (and
    any other schema used here) have all-default fields, so a no-arg construction
    always succeeds; guarded with model_construct() as a last-resort fallback."""
    try:
        return schema_model()
    except Exception:
        return schema_model.model_construct()


def _example_value(annotation: Any, field_name: str, depth: int) -> Any:
    """One plausible example value for a single field's type annotation, recursing into
    nested BaseModel/list types. Not a rigorous type-system walker (Literal/Union beyond
    Optional fall back to a placeholder string) — good enough for this project's own
    schemas (str/int/float/bool/list/nested-BaseModel), which is all `_example_instance`
    below needs to handle."""
    if depth > 3:
        return None
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        item_type = args[0] if args else str
        if isinstance(item_type, type) and issubclass(item_type, BaseModel):
            return [_example_instance(item_type, depth + 1)]
        return [_example_value(item_type, field_name, depth + 1)]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _example_instance(annotation, depth + 1)
    if annotation is int:
        return 1
    if annotation is float:
        return 1.0
    if annotation is bool:
        return True
    return f"example_{field_name}"


def _example_instance(schema_model: Type[BaseModel], depth: int = 0) -> dict:
    """Build a plausible, NON-EMPTY example instance (as a plain dict, ready for
    `json.dumps`) for schema_model, recursively for nested BaseModel/list fields."""
    return {name: _example_value(f.annotation, name, depth)
            for name, f in schema_model.model_fields.items()}


def _schema_instruction(schema_model: Type[BaseModel]) -> str:
    """Prompt suffix telling the model the exact JSON shape to return.

    Uses a concrete WORKED EXAMPLE (a plausible non-empty instance), not a raw JSON-schema
    dump. Empirically, Qwen3-VL-8B responds to an abstract schema dump by returning an
    empty-but-valid instance (e.g. `{"tags": []}`) instead of doing the actual extraction
    task, even with a generous token budget — GPT-5.5-low tolerates the schema-dump style
    fine, so this was invisible until testing the local Qwen path directly (diagnosed live,
    2026-07-23, in `notebooks/e2e_harness/PartB_Qwen_RelationRun_GPUOnly.ipynb`: schema dump
    + 4096 tokens -> empty in 3.8s; schema dump + 32000 tokens -> still empty in 3.8s, so it
    was never a token-budget issue; the SAME 4096-token budget with a worked example instead
    of the schema dump produced real, correct content). Falls back to the old schema-dump
    behavior only if example generation ever fails for an unusual schema shape (defensive,
    not expected to trigger for this project's own schemas)."""
    try:
        example_json = json.dumps(_example_instance(schema_model))
        return (
            "\n\nRespond with a SINGLE JSON object, no prose outside it, in exactly this "
            "shape (a worked EXAMPLE of the fields/structure, not literal content to "
            f"copy):\n{example_json}\n\n"
            "Fill in REAL content from the task above — do not return empty lists or "
            "blank strings unless the source genuinely has nothing to report there."
        )
    except Exception:
        try:
            schema_json = json.dumps(schema_model.model_json_schema())
        except Exception:
            schema_json = "{}"
        return (
            "\n\nRespond with a SINGLE JSON object in a ```json fenced code block that "
            "strictly conforms to this JSON Schema. No prose outside the fence.\n"
            f"JSON Schema:\n{schema_json}"
        )


def _iter_balanced_objects(text: str):
    """Yield every COMPLETE, well-formed `{...}` substring of `text`, scanning from
    every `{` occurrence (not just top-level ones) with a proper string-aware
    brace-depth scanner (so braces/brackets inside string VALUES -- e.g. a tag
    text containing a literal `{` -- never confuse the match). This is a balanced-
    brace scanner, not a naive regex, precisely so nested objects (a real
    `OcrTag` living inside a truncated `OcrResult.tags` array) are still found even
    though the enclosing object/array never closes.

    Candidates are yielded in ascending start-position order, so complete items
    that appear before the truncation point come out before it (and anything
    after the last full `}` -- the truncated tail -- never yields at all, since
    its brace never finds a matching close)."""
    n = len(text)
    for start in range(n):
        if text[start] != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, n):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    break


def _list_of_submodel_field(schema_model: Type[BaseModel]):
    """If `schema_model` has exactly one field typed `List[SomeBaseModel]` (e.g.
    `OcrResult.tags: List[OcrTag]`), return (field_name, SomeBaseModel). Otherwise
    None. Generic over pydantic v2 `model_fields` -- no hardcoded schema names, so
    it works for `OcrResult`/`OcrTag` today without hardcoding either name."""
    try:
        fields = schema_model.model_fields
    except Exception:
        return None
    for name, f in fields.items():
        ann = f.annotation
        if get_origin(ann) is list:
            args = get_args(ann)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                return name, args[0]
    return None


def _salvage_partial(raw_text: str, schema_model: Type[BaseModel]) -> Optional[BaseModel]:
    """Companion safety fix (Phase 1c item 3): called only after the normal
    `_extract_json_text` + `json.loads` + `model_validate` path already failed
    (truncated/broken JSON, e.g. from a capped `max_new_tokens`). Recovers whatever
    COMPLETE list items it can, rather than unconditionally returning an empty
    default instance -- matching the real pipeline's own convention
    (`llm_proxy._salvage_json`, `ocr_reasoning.py`'s docstring: "partial > nothing").

    Only handles schemas with a `List[BaseModel]` field (the real case that
    matters here: `OcrResult.tags: List[OcrTag]`). Schemas without one (`_OneTag`,
    `_ManyTags` -- `List[str]`, not a sub-model) have nothing this step can safely
    salvage, so it returns None and the caller falls back to the empty default,
    unchanged from before this fix."""
    list_field = _list_of_submodel_field(schema_model)
    if list_field is None:
        return None
    field_name, item_model = list_field

    items: List[BaseModel] = []
    for candidate in _iter_balanced_objects(raw_text):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        try:
            items.append(item_model.model_validate(parsed))
        except Exception:
            continue

    if not items:
        return None

    try:
        return schema_model(**{field_name: items})
    except Exception:
        try:
            return schema_model.model_construct(**{field_name: items})
        except Exception:
            return None


def build_qwen_call_llm(
    generate_fn: GenerateFn,
    *,
    max_new_tokens_cap: Optional[int] = None,
) -> Callable[..., Any]:
    """Build the local `call_llm` callable. `generate_fn` is called SYNCHRONOUSLY
    (Qwen3-VL generation is not natively async) under a lock so concurrent callers
    serialize cleanly rather than racing a single model instance.

    `max_new_tokens_cap`: optional Phase-1c plumbing (no real Qwen wrapper exists
    yet -- that's Phase 2, GPU). When set, it is threaded through to `generate_fn`
    as a keyword argument (`generate_fn(prompt, images, max_new_tokens=...)`) so a
    real Qwen3-VL wrapper can honor a hard generation-length cap (latency fix,
    Extraction_Agent_Local_Plan.md §11 Phase 1c item 3) without this interface
    needing to change again in Phase 2. This is a hard CEILING override for
    latency-tuning -- it takes priority over the caller's own per-call `max_tokens`
    even when that's larger, by design.

    Precedence when calling `generate_fn` (bug found + fixed 2026-07-23: the
    per-call `max_tokens` argument used to be silently dropped whenever no
    `max_new_tokens_cap` was configured, falling back to whatever hardcoded
    default `generate_fn` itself happened to have -- e.g. 4096 -- regardless of
    what the real pipeline asked for. On a dense sheet needing hundreds of tags,
    prod's own `max_tokens=32000` call got silently capped at 4096, truncating
    the JSON mid-array and producing 0 tags with no visible error, since the
    parse failure below is caught and swallowed too):
      1. `max_new_tokens_cap` if configured (hard ceiling, wins even over a
         larger per-call `max_tokens`).
      2. else the caller's own per-call `max_tokens`, if given.
      3. else `generate_fn` is called with neither, falling back to whatever
         default `generate_fn` itself was built with."""
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
        model_name = model or "qwen3-vl-8b-local"
        full_prompt = prompt + _schema_instruction(schema_model)
        imgs = list(images or [])

        async with lock:
            if max_new_tokens_cap is not None:
                raw_text = generate_fn(full_prompt, imgs, max_new_tokens=max_new_tokens_cap)
            elif max_tokens is not None:
                raw_text = generate_fn(full_prompt, imgs, max_new_tokens=max_tokens)
            else:
                raw_text = generate_fn(full_prompt, imgs)

        slot = usage.setdefault(
            model_name, {"in": 0, "cache_w": 0, "cache_r": 0, "out": 0, "calls": 0}
        )
        slot["calls"] += 1

        try:
            json_text = _extract_json_text(raw_text or "")
            if json_text is None:
                raise ValueError("no JSON-like content found in generate_fn output")
            parsed = json.loads(json_text)
            return schema_model.model_validate(parsed)
        except Exception:
            # Companion safety fix (Phase 1c item 3): a hard max_new_tokens cap
            # increases truncation risk, so before giving up entirely, try to
            # salvage whatever complete tag objects the raw output DID finish --
            # matching the real pipeline's own convention (llm_proxy.py's
            # `_salvage_json`/ocr_reasoning.py's docstring: "partial > nothing").
            salvaged = _salvage_partial(raw_text or "", schema_model)
            if salvaged is not None:
                return salvaged
            # Honest failure mode: empty/default instance, matching the pipeline's
            # own convention (grounded_read.py's `read_shapes` already treats an
            # empty `.tag` as "no tag found", not a crash).
            return _default_instance(schema_model)

    call_llm.usage = usage  # type: ignore[attr-defined]
    return call_llm
