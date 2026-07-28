import asyncio

import pytest
from pydantic import BaseModel

from extraction_local.qwen_call_llm import build_qwen_call_llm


class _OneTag(BaseModel):
    tag: str = ""


class _ManyTags(BaseModel):
    tags: list = []


def test_one_tag_success():
    def fake_generate(prompt, images):
        assert isinstance(prompt, str) and prompt
        return '```json\n{"tag": "TEST-101"}\n```'

    call_llm = build_qwen_call_llm(fake_generate)
    out = asyncio.run(call_llm("read this", _OneTag, images=["b64data"], model="qwen3-vl-8b", max_tokens=40))
    assert out.tag == "TEST-101"


def test_garbage_output_falls_back_to_empty_tag():
    def fake_generate(prompt, images):
        return "not json at all, sorry"

    call_llm = build_qwen_call_llm(fake_generate)
    out = asyncio.run(call_llm("read this", _OneTag, images=["b64"], model="qwen3-vl-8b", max_tokens=40))
    assert out.tag == ""


def test_many_tags_multi_image_call():
    seen_images = {}

    def fake_generate(prompt, images):
        seen_images["images"] = images
        return '{"tags": ["TEST-102", "TEST-103"]}'

    call_llm = build_qwen_call_llm(fake_generate)
    out = asyncio.run(
        call_llm("read region", _ManyTags, images=["b64a", "b64b"], model="qwen3-vl-8b", max_tokens=120)
    )
    assert out.tags == ["TEST-102", "TEST-103"]
    assert seen_images["images"] == ["b64a", "b64b"]


def test_usage_dict_shape():
    def fake_generate(prompt, images):
        return '{"tag": "X-1"}'

    call_llm = build_qwen_call_llm(fake_generate)
    asyncio.run(call_llm("p", _OneTag, images=[], model="my-model", max_tokens=40))
    asyncio.run(call_llm("p", _OneTag, images=[], model="my-model", max_tokens=40))
    assert "my-model" in call_llm.usage
    slot = call_llm.usage["my-model"]
    assert set(slot.keys()) == {"in", "cache_w", "cache_r", "out", "calls"}
    assert slot["calls"] == 2


def test_default_model_used_when_none_given():
    def fake_generate(prompt, images):
        return '{"tag": "X-1"}'

    call_llm = build_qwen_call_llm(fake_generate)
    asyncio.run(call_llm("p", _OneTag, images=[]))
    assert len(call_llm.usage) == 1


# --------------------------------------------------------------------------- #
# Phase 1c fixes
# --------------------------------------------------------------------------- #

class OcrTag(BaseModel):
    text: str = ""
    type: str = ""
    word_ids: list = []
    bbox: list = []


class OcrResult(BaseModel):
    standard: str = ""
    prefix: str = ""
    tags: list[OcrTag] = []


def test_max_new_tokens_cap_threaded_through_to_generate_fn():
    seen = {}

    def fake_generate(prompt, images, max_new_tokens=None):
        seen["max_new_tokens"] = max_new_tokens
        return '{"tag": "X-1"}'

    call_llm = build_qwen_call_llm(fake_generate, max_new_tokens_cap=800)
    asyncio.run(call_llm("p", _OneTag, images=[]))
    assert seen["max_new_tokens"] == 800


def test_no_cap_calls_fake_with_original_two_arg_signature():
    # No cap configured -> generate_fn must be called exactly as before (2 args),
    # so existing fakes with the old signature keep working unmodified.
    def fake_generate(prompt, images):
        return '{"tag": "X-1"}'

    call_llm = build_qwen_call_llm(fake_generate)  # no max_new_tokens_cap
    out = asyncio.run(call_llm("p", _OneTag, images=[]))
    assert out.tag == "X-1"


def test_salvage_recovers_complete_tags_from_truncated_ocr_result():
    # Deliberately truncated OcrResult JSON: 2 complete OcrTag objects, then a
    # partial/cut-off third one, no closing brackets at all (simulates a real
    # max_tokens truncation mid-array).
    truncated = (
        '{"standard": "ISA-5.1", "prefix": "", "tags": ['
        '{"text": "FT-101", "type": "instrument", "word_ids": [1, 2], "bbox": []}, '
        '{"text": "PV-202", "type": "valve", "word_ids": [5], "bbox": []}, '
        '{"text": "TI-30'
    )

    def fake_generate(prompt, images):
        return truncated

    call_llm = build_qwen_call_llm(fake_generate)
    out = asyncio.run(call_llm("p", OcrResult, images=["b64"], model="qwen3-vl-8b", max_tokens=800))

    assert len(out.tags) == 2, out.tags
    texts = {t.text for t in out.tags}
    assert texts == {"FT-101", "PV-202"}


def test_salvage_returns_default_when_no_list_of_submodel_field():
    # _OneTag/_ManyTags have no List[BaseModel] field -- salvage has nothing safe
    # to recover, so garbage output still falls back to the honest empty default,
    # unchanged from before this fix.
    def fake_generate(prompt, images):
        return 'not json, and no salvageable objects either'

    call_llm = build_qwen_call_llm(fake_generate)
    out = asyncio.run(call_llm("p", _OneTag, images=[]))
    assert out.tag == ""
