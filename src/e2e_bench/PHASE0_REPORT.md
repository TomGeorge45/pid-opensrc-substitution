# Phase 0 Report — Environment + Import Verification

**Date:** 2026-07-16
**Verdict: no blockers found.** Everything the plan flagged as risky turned out importable
and injectable. All 5 "open items" in `Conversion_Layer_Plan.md` §9 are resolved below.

## Environment

- Agent requires Python ≥3.10; system `/usr/bin/python3` is 3.9.6 — **not sufficient**.
- Used pyenv's `3.12.10` (`~/.pyenv/versions/3.12.10/bin/python3`) to create
  `pid-ml/.venv-e2e`. All subsequent work in this venv (`source .venv-e2e/bin/activate`).
- Editable-installed, in order (each resolved clean):
  1. `rive-ai-platform/shared/entity_operations`
  2. `rive-ai-platform/shared/security` (declares itself as PyPI name `rive-security` —
     `rive_adk` depends on `rive-security>=0.1.0`, which doesn't exist on PyPI; this monorepo
     package satisfies it once installed by path)
  3. `rive-ai-platform/shared/rive_adk`
  4. `rive-ai-platform/agents/pnid-intelligence-agent` itself (installs `pnid_agent` as an
     importable package, pulling in its PyPI deps: pydantic≥2, opencv-python, numpy,
     scikit-image, pymupdf, anthropic, etc.)
- No `constraints-e2e.txt` pins needed beyond what pip resolved automatically — no version
  conflicts surfaced. (Plan §3 asked to record exact resolved versions if pinning was
  needed; it wasn't.)

## Import checklist (plan §3.2) — ALL PASS

Every module listed in the plan imported cleanly, including the two flagged as likely to
fail:

- `pnid_agent.sub_agents.symbol_detection.driver` (needed for D6's preferred real
  `_compose_detection_records`) — **imports clean**. D6 fallback (hand-replicating the
  function) is NOT needed.
- `pnid_agent.stages.graph_construction.driver`, `pnid_agent.sub_agents.entity_validation.driver`,
  `pnid_agent.sub_agents.relation_validation.driver` (need `rive_adk`) — **import clean**.

One non-fatal logged error appears on every import that touches `rive_adk`:
```
ERROR:rive_adk.core.config:Configuration validation failed: 1 validation error for AgentConfig
agent
  Field required [type=missing, input_value={}, input_type=dict]
```
This does not raise or block import — it's `rive_adk` attempting some default config load
at import time and logging (not raising) on failure. Not yet exercised at call time (we
haven't invoked `stage_11_run` etc. with real args yet — only introspected signatures).
**Flag to re-check** once the harness actually calls these drivers: if it turns out to
matter, it likely wants a real `AgentConfig`-shaped env/config file present, or is truly
inert. Two harmless `SyntaxWarning`s also appear (`prompt.py:92,556`, unescaped `\|` in a
docstring table) — cosmetic, ignore.

## Open items from the plan, resolved

**1. Does `sub_agents/symbol_detection/driver.py` import cleanly?** Yes. D6 fallback not needed.

**2. Do `stage_13_run`/`stage_12_run` accept an injectable runner cleanly?** Yes, via
`vlm_runner: Optional[Any] = None` on both. BUT the injection point is lower-level than
"give me an answer" — the driver calls `runner._get_messages_client()` then talks to that
client using the raw Anthropic `messages.create(model=..., tools=[...], tool_choice=...,
messages=[...])` shape directly (same pattern already found in stage 4's `detector.py`).
**Confirmed the response-parsing code on both stages is defensive** (`getattr(obj, "attr",
None)` with dict `.get()` fallback — `entity_validation/driver.py:586-596`,
`relation_validation/relation_validator.py` `_extract_tool_use`/payload access) — so the
fake response can be **plain dicts / `SimpleNamespace`**, not real `anthropic` SDK types.

  **Concrete design (supersedes plan's tentative (a)/(b) framing — (a) is confirmed and
  it's simple):** one shared `FakeMessagesClient` in `e2e_bench/assembly/fake_llm.py`:
  ```python
  class FakeMessagesClient:
      def __init__(self, next_answer_fn):  # next_answer_fn() -> dict payload for the tool
          self._next = next_answer_fn
      class _Messages:
          def __init__(self, outer): self._outer = outer
          async def create(self, **kwargs):
              payload = self._outer._next(kwargs)  # harness maps call -> precomputed answer
              return SimpleNamespace(
                  content=[{"type": "tool_use", "name": kwargs["tool_choice"]["name"], "input": payload}],
                  usage={"input_tokens": 0, "output_tokens": 0},
              )
      @property
      def messages(self): return self._Messages(self)

  class FakeRunner:
      def __init__(self, client): self._client = client
      def _get_messages_client(self): return self._client
  ```
  `next_answer_fn(kwargs)` is where the harness correlates "which entity/relation is this
  call about" — either by inspecting `kwargs["messages"]` text for an id the harness
  embedded, or (simpler, since call order is deterministic and single-threaded per page in
  stage 13; **stage 12 fans out concurrently via `asyncio.gather`**, so stage 12's
  `next_answer_fn` MUST correlate by content, not call order — e.g. parse the relation_id
  out of the prompt text, or thread it via a per-call closure built by the harness from the
  known relation list before dispatch).

**3. Exact `rive_adk`/`entity_operations` versions?** Both installed via editable path
install from the monorepo — no PyPI version resolution involved, so "version" = whatever's
checked out in the monorepo working tree at `rive-ai-platform/shared/{entity_operations,rive_adk}`
right now. No lockfile exists; nothing to pin beyond "same monorepo checkout."

**4. Does `stage_06_run` write its own output?** Not yet directly verified by execution
(only import-checked) — confirm during Phase 2 (§5, stage 6 sequencing task) by actually
running it on a real page and checking `stage-06/stage_06_output.json` appears. Expected
per the plan's citation of `line_tracing/driver.py`; treat as high-confidence, not proven.

**5. Is `stage_11_run`'s config surface practical?** Yes — try the real driver first.
38 parameters, all but the first 3 (`context`, `artifact_store`, `drawing_document`) have
defaults. `ontology_payload_factory` and `token_provider=None` are exactly the injection
points the plan hoped for (D2). `context: Any` is treated permissively everywhere observed
(`getattr(context, "tenant_id", None) or "default"`) — a bare `SimpleNamespace(tenant_id=
"benchmark")` satisfies every usage found so far. No hand-chaining fallback needed;
proceed with the real driver.

## Net effect on the plan

- D6 (fallback compose function) — **not needed**, delete from critical path, keep as a
  documented "if this ever breaks" note only.
- §5.6/5.7 "(a) vs (b)" — **(a) confirmed**, and simpler than anticipated (shared
  `FakeMessagesClient`, ~20 lines, not two separate reimplementations of stage 13/12's
  write logic). This meaningfully de-risks and shrinks the remaining build.
- One real new sub-task surfaced: stage 12's concurrent dispatch means the fake client's
  answer-correlation must be content-based (relation_id-aware), not order-based. Stage 13's
  can be simpler (sequential per page) but should use the same content-based approach for
  consistency and to avoid a subtle ordering bug if stage 13's concurrency model changes.

## Environment reproduction (for the executor / future sessions)

```bash
cd /Users/tomgeorge/pid-ml
~/.pyenv/versions/3.12.10/bin/python3 -m venv .venv-e2e
source .venv-e2e/bin/activate
pip install -e /Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/shared/entity_operations
pip install -e /Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/shared/security
pip install -e /Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/shared/rive_adk
pip install -e /Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/agents/pnid-intelligence-agent
```
