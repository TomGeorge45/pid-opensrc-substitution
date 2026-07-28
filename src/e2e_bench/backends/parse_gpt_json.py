"""GPT-5.5 backend re-export — see parse_json_common.py (D9: one shared JSON conversion path
for both Qwen and GPT-5.5, since they both emit JSON of the same shapes)."""
from .parse_json_common import (  # noqa: F401
    parse_entity_verdict_json,
    parse_relation_verdict_json,
    parse_skid_json,
    parse_titleblock_json,
)
