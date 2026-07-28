"""
Fake Anthropic-messages-shaped client, used to drive the REAL stage_13_run/stage_12_run
drivers with pre-computed local/GPT model answers instead of live Anthropic calls.

Confirmed in PHASE0_REPORT.md: both drivers call `runner._get_messages_client()` then talk
to it via the raw `client.messages.create(model=..., tools=[...], tool_choice=...,
messages=[...])` shape, and their response-parsing is defensive (getattr with dict
fallback — entity_validation/driver.py:586-596, relation_validation/relation_validator.py),
so the fake response can be plain dicts/SimpleNamespace, not real `anthropic` SDK types.

IMPORTANT (PHASE0_REPORT.md open item, resolved): stage 13 processes entities sequentially
per page, but stage 12 dispatches concurrently via `asyncio.gather` — so the answer lookup
MUST correlate by content (an id embedded in the call), never by call order.
"""
from types import SimpleNamespace


class FakeResponse(SimpleNamespace):
    """Duck-types enough of an anthropic.types.Message for _extract_tool_payload /
    _extract_tool_use and _usage_dict / usage_dict to work (all getattr-with-dict-fallback)."""
    pass


class FakeMessagesClient:
    """`next_answer_fn(kwargs) -> dict` receives the exact kwargs passed to
    `messages.create(...)` (model, max_tokens, system, tools, tool_choice, messages) and
    must return the tool-call payload dict for whatever entity/relation this call is about.
    The caller is responsible for building `next_answer_fn` so it can identify which
    entity/relation a given call is for — typically by finding a known id (temp_id or
    relation_id) as a substring somewhere in `kwargs["messages"]`'s text content, since the
    harness already knows, before dispatch, exactly which entities/relations will be asked
    about and in what call shape."""

    def __init__(self, next_answer_fn):
        self._next_answer_fn = next_answer_fn
        self.calls = []  # recorded for debugging/tests

    @property
    def messages(self):
        return _FakeMessagesNamespace(self)


class _FakeMessagesNamespace:
    def __init__(self, outer: FakeMessagesClient):
        self._outer = outer

    async def create(self, **kwargs):
        self._outer.calls.append(kwargs)
        tool_name = kwargs["tool_choice"]["name"]
        payload = self._outer._next_answer_fn(kwargs)
        return FakeResponse(
            content=[{"type": "tool_use", "name": tool_name, "input": payload}],
            usage={"input_tokens": 0, "output_tokens": 0,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        )


class FakeRunner:
    def __init__(self, client: FakeMessagesClient):
        self._client = client

    def _get_messages_client(self):
        return self._client
