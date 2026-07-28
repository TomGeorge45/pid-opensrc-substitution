"""Gap #12 — standalone shim for pipelines 1/2's REAL relation code.

Runs the actual, unmodified `pnid_pipeline.hierarchy.apply_hierarchy` (prod's LLM-based
relation/hierarchy pass) on entities we inject, with a swappable `call_llm` — the exact
proven pattern from `src/extraction_local/run_extraction_local.py::run_one_sheet`, which
already does this for the extraction pass (real `extract_page`, injected `call_llm`, zero
agent-source edits). Nothing here reimplements prod logic; `apply_hierarchy` runs verbatim.

Swappable call_llm — both already exist, no new client code needed:
  - Arm 1 (prod, GPT-5.5-low): `pnid_pipeline.llm_proxy.build_call_llm(model="gpt-5.5-low")`
    — confirmed by reading llm_proxy.py directly: the proxy's own comment states
    "gpt-5.5 (openai_proxy) ... untouched" alongside Sonnet 4.6, i.e. the LiteLLM-style
    proxy already routes a gpt-5.5 model name through this same Anthropic-Messages-shaped
    `/v1/messages` call — the real prod contract `apply_hierarchy` expects
    (`call_llm(prompt, schema_model, *, images=, model=, max_tokens=)` -> BaseModel).
  - Arm 2 (local): `extraction_local.qwen_call_llm.build_qwen_call_llm(generate_fn)` — same
    contract, already built and used by `run_one_sheet` for the extraction pass.

IMPORTANT — the entity-injection gap this shim can't paper over: PID2Graph GT nodes carry
no tag TEXT (class-agnostic, like Gupta — CLAUDE.md rule 5), but prod's hierarchy pass is
built entirely around tag-text reasoning (`_nesting_parent`'s prefix matching, ISA loop
hints, the LLM prompt's `id|TEXT|type|cx,cy` lines). Feeding it text-less entities means:
  - `_nesting_parent` returns {} (its alnum filter drops every empty-text tag).
  - The LLM prompt's per-tag line degrades to `id||type|cx,cy` — the model has almost
    nothing but a class label and a position to reason from.
This is an HONEST test of a real, reportable degraded-input scenario (arms 1/2 face the
exact same text-less-ness that PID2Graph's dataset design imposes on every arm), not a bug
in the shim. It is exactly why AG/RIVE annotated fixture sheets (Group 2, gap #3+5) matter:
only there do arms 1/2's hierarchy pass get to reason over real tag text.
"""
from __future__ import annotations

import sys
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

AGENT_DIR = "/Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/agents/pnid-extraction-agent"
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

import numpy as np  # noqa: E402

from pnid_pipeline.hierarchy import apply_hierarchy  # noqa: E402
from pnid_pipeline.models import DrawingMeta, ExtractionResult, Tag  # noqa: E402

from pid2graph_gt import GTNode  # noqa: E402
from type_vocab import to_extraction_agent_type  # noqa: E402

DEFAULT_CFG: Dict[str, Any] = {"hierarchy_pass": {"max_missed": 0, "add_missed": False,
                                                   "connectivity": True}}


def _tags_from_gt_nodes(nodes: Dict[str, GTNode], page_w: int, page_h: int) -> List[Tag]:
    """Build prod-shaped `Tag` objects from PID2Graph GT nodes. `text=""` throughout —
    see module docstring for why, and what that costs the real pass. `type` goes through
    the frozen gap-#7 mapping (`type_vocab.py`), not PID2Graph's raw class name — see that
    module's docstring for why an un-mapped class would silently bias this arm."""
    tags: List[Tag] = []
    for nid, n in nodes.items():
        x0, y0, x1, y1 = n.bbox
        bbox_norm = [x0 / page_w, y0 / page_h, x1 / page_w, y1 / page_h]
        cx, cy = n.center
        tags.append(Tag(
            id=nid, text="", raw_text="", type=to_extraction_agent_type(n.cls),
            bbox_px=[x0, y0, x1, y1], bbox_norm=bbox_norm, center_px=[cx, cy],
            source="ground_truth", confidence=1.0,
        ))
    return tags


async def run_hierarchy_shim(
    nodes: Dict[str, GTNode],
    page_gray: np.ndarray,
    call_llm: Callable,
    *,
    model: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> ExtractionResult:
    """Run the REAL `apply_hierarchy` on injected GT entities. `page_gray` must be a plain
    numpy array (grayscale or BGR) — `apply_hierarchy` -> `_overview` -> `_overview_b64`
    expects `img.shape[:2]`, the same contract `ocr_reasoning_extract` uses in prod."""
    h, w = page_gray.shape[:2]
    tags = _tags_from_gt_nodes(nodes, w, h)
    result = ExtractionResult(
        drawing=DrawingMeta(source_file="relation_bench_shim", canvas_px=(w, h)),
        tags=tags,
    )
    return await apply_hierarchy(result, page_gray, w, h, call_llm, model, cfg or DEFAULT_CFG)


def topology_pairs_from_result(result: ExtractionResult) -> Set[FrozenSet[str]]:
    """Extract undirected (parent, child) pairs from the mutated tags' `parent_id`, in the
    same shape `score.py::score_topology` expects — so pipelines 1/2's real hierarchy
    output is scoreable with the exact same harness as R1/R2a/R2b (pipeline 3)."""
    pairs: Set[FrozenSet[str]] = set()
    for t in result.tags:
        if t.parent_id:
            pairs.add(frozenset((t.id, t.parent_id)))
    return pairs
