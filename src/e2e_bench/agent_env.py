"""
Locates the real pnid-intelligence-agent installation and verifies it's importable.

The agent package (`pnid_agent`) plus its monorepo-shared dependencies (`entity_operations`,
`rive_adk`, `rive_security`) must already be editable-installed into the active venv — see
PHASE0_REPORT.md for the exact install commands. This module does not install anything; it
only locates the agent's source root (for reference/logging) and provides one function that
fails fast with a clear message if the required packages aren't importable, instead of a
confusing import error deep inside a converter.
"""
import importlib
import os

DEFAULT_AGENT_REPO = (
    "/Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform/agents/pnid-intelligence-agent"
)

REQUIRED_MODULES = (
    "pnid_agent.models.page_classification",
    "pnid_agent.models.page_ocr",
    "pnid_agent.models.title_block",
    "pnid_agent.models.detections",
    "pnid_agent.models.line_tracing",
    "pnid_agent.models.rive_ontology",
    "pnid_agent.models.ontology_mapping",
    "pnid_agent.models.relation_validation",
    "pnid_agent.models.drawing_document",
    "pnid_agent.models.provenance",
    "pnid_agent.shared.coord_ops",
    "pnid_agent.stages.tile_segmentation.grid",
    "pnid_agent.stages.tile_segmentation.exclusion",
    "pnid_agent.sub_agents.symbol_detection.nms",
    "pnid_agent.stages.line_tracing.driver",
    "pnid_agent.stages.graph_construction.relations",
    "pnid_agent.stages.graph_construction.ontology_relation_index",
    "pnid_agent.stages.graph_construction.inference",
    "pnid_agent.stages.graph_construction.entities",
    "pnid_agent.sub_agents.symbol_detection.driver",
    "pnid_agent.stages.graph_construction.driver",
    "pnid_agent.sub_agents.entity_validation.driver",
    "pnid_agent.sub_agents.relation_validation.driver",
    "pnid_agent.storage.local_fs",
)


def agent_repo_path() -> str:
    return os.environ.get("PNID_AGENT_REPO", DEFAULT_AGENT_REPO)


def verify_agent_importable() -> None:
    """Raise a clear, actionable RuntimeError if the agent isn't importable, rather than
    letting a converter fail with a confusing traceback deep in its own logic."""
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as e:
            missing.append(f"{mod}: {type(e).__name__}: {e}")
    if missing:
        raise RuntimeError(
            "e2e_bench requires the pnid-intelligence-agent package (and its monorepo "
            "dependencies entity_operations/rive_adk/rive_security) to be editable-installed "
            "into the active venv. See src/e2e_bench/PHASE0_REPORT.md for exact commands.\n"
            "Failed imports:\n  " + "\n  ".join(missing)
        )
