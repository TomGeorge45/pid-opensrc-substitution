"""
Shared JSON-answer extraction + per-answer-type parsing, used by BOTH Qwen and GPT-5.5
backends (D9: one conversion path for both arms — they emit the same JSON shapes, only the
raw text differs). parse_qwen_json.py and parse_gpt_json.py are thin re-exports of this
module; keep backend-specific quirks (if any ever appear) in those files, not here.

Extraction strategy, in order: fenced ```json ... ``` block (last one wins, matching the
pattern used throughout this project's benchmark notebooks) -> first balanced brace/bracket
span -> fail.
"""
import json
import re

from ..types import (
    NormalizedEntityVerdict,
    NormalizedRelationVerdict,
    NormalizedSkidAssignment,
    NormalizedSkidMember,
    NormalizedTitleBlock,
    ParseOutcome,
)

_FENCED_RE = re.compile(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", re.S)


def _extract_json_text(text: str):
    """Returns the raw JSON substring to attempt, or None if nothing plausible found."""
    fenced = _FENCED_RE.findall(text)
    if fenced:
        return fenced[-1]
    # loose fallback: whichever opening bracket ({ or [) appears FIRST in the text wins -
    # trying "{" unconditionally before "[" would wrongly truncate a top-level JSON array
    # of objects down to just its first nested object (caught by parse_skid_json's test).
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append((start, text[start:end + 1]))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def _parse_json(text: str):
    """-> (parsed_value, error_str_or_None). Never raises."""
    candidate = _extract_json_text(text)
    if candidate is None:
        return None, "no JSON-like content found"
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"


TB_FIELDS = ["drawing_number", "revision", "title", "site"]


def parse_titleblock_json(text: str) -> ParseOutcome:
    """Expects {"drawing_number":..., "revision":..., "title":..., "site":...} (any subset;
    missing keys -> None). D5: these are the benchmark's declared tenant-schema field names."""
    parsed, err = _parse_json(text)
    if parsed is None or not isinstance(parsed, dict):
        return ParseOutcome.failed(text, err or "parsed JSON is not an object")
    fields = {k: parsed.get(k) for k in TB_FIELDS}
    located = any(v for v in fields.values())
    return ParseOutcome.ok(NormalizedTitleBlock(located=located, fields=fields), text)


def parse_skid_json(text: str, asset_temp_id: str) -> ParseOutcome:
    """Expects a JSON list of member proposals:
    [{"target_temp_id": "...", "forward_relation_name": "..."|null,
      "confidence": 0.9, "reasoning": "..."}, ...]"""
    parsed, err = _parse_json(text)
    if parsed is None or not isinstance(parsed, list):
        return ParseOutcome.failed(text, err or "parsed JSON is not a list")
    members = []
    for item in parsed:
        if not isinstance(item, dict) or "target_temp_id" not in item:
            continue
        members.append(NormalizedSkidMember(
            target_temp_id=str(item["target_temp_id"]),
            forward_relation_name=item.get("forward_relation_name"),
            confidence=float(item.get("confidence", 0.0)),
            reasoning=str(item.get("reasoning", "")),
        ))
    return ParseOutcome.ok(NormalizedSkidAssignment(asset_temp_id=asset_temp_id, members=members), text)


def parse_entity_verdict_json(text: str, temp_id: str) -> ParseOutcome:
    """Expects {"keep": true|false, "confidence": 0.9, "reasoning": "..."}. Falls back to
    plain "keep"/"remove" text (case-insensitive substring) if JSON extraction fails, since
    the v3-stage13 adapter's task format (this project's own benchmark) was trained on
    exactly that plain-text keep/remove answer, not JSON."""
    parsed, err = _parse_json(text)
    if isinstance(parsed, dict) and "keep" in parsed:
        return ParseOutcome.ok(NormalizedEntityVerdict(
            temp_id=temp_id, keep=bool(parsed["keep"]),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=str(parsed.get("reasoning", "")),
        ), text)
    lowered = text.strip().lower()
    has_keep = re.search(r"\bkeep\b", lowered) is not None
    has_remove = re.search(r"\bremove\b", lowered) is not None
    if has_remove and not has_keep:
        return ParseOutcome.ok(NormalizedEntityVerdict(temp_id=temp_id, keep=False, confidence=0.5), text)
    if has_keep and not has_remove:
        return ParseOutcome.ok(NormalizedEntityVerdict(temp_id=temp_id, keep=True, confidence=0.5), text)
    return ParseOutcome.failed(text, err or "no JSON object with 'keep' and no unambiguous plain keep/remove text found")


_VALID_VERDICTS = {"confirmed", "rejected", "uncertain"}


def parse_relation_verdict_json(text: str, relation_id: str) -> ParseOutcome:
    """Expects {"verdict": "confirmed"|"rejected"|"uncertain", "revised_confidence": 0.9,
    "reasoning": "..."}. Falls back to plain yes/no text (this project's own v3-relation
    benchmark task format) mapped yes->confirmed, no->rejected."""
    parsed, err = _parse_json(text)
    if isinstance(parsed, dict) and parsed.get("verdict") in _VALID_VERDICTS:
        return ParseOutcome.ok(NormalizedRelationVerdict(
            relation_id=relation_id, verdict=parsed["verdict"],
            revised_confidence=float(parsed.get("revised_confidence", 0.5)),
            reasoning=str(parsed.get("reasoning", "")),
            suggested_alternative_relation_name=parsed.get("suggested_alternative_relation_name"),
        ), text)
    lowered = text.strip().lower()
    has_yes = re.search(r"\byes\b", lowered) is not None
    has_no = re.search(r"\bno\b", lowered) is not None
    if has_yes and not has_no:
        return ParseOutcome.ok(NormalizedRelationVerdict(
            relation_id=relation_id, verdict="confirmed", revised_confidence=0.75), text)
    if has_no and not has_yes:
        return ParseOutcome.ok(NormalizedRelationVerdict(
            relation_id=relation_id, verdict="rejected", revised_confidence=0.25), text)
    return ParseOutcome.failed(text, err or "no valid verdict JSON and no unambiguous plain yes/no text found")
