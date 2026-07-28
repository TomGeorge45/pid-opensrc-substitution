"""
Loads the frozen benchmark ontology (benchmark_ontology.json) into the real agent's
OntologyRelationIndex, plus an entity_type -> human name lookup for detection assembly.

See benchmark_ontology.json's "_comment" for why this exists and its coverage caveat.
"""
import json
from pathlib import Path

from pnid_agent.stages.graph_construction.ontology_relation_index import OntologyRelationIndex

_ONTOLOGY_PATH = Path(__file__).parent / "benchmark_ontology.json"


def load_benchmark_ontology_raw() -> dict:
    with open(_ONTOLOGY_PATH) as f:
        return json.load(f)


def load_ontology_relation_index() -> OntologyRelationIndex:
    raw = load_benchmark_ontology_raw()
    return OntologyRelationIndex.from_entries(raw["relations"])


def entity_type_names() -> dict:
    """entity_type (semanticId) -> human-readable name. Since the benchmark ontology's
    entity_types are already human-readable-ish strings, this is currently identity with
    light formatting; kept as a separate function so a richer name map can be dropped in
    without touching call sites."""
    raw = load_benchmark_ontology_raw()
    return {t: t.replace("_", " ").title() for t in raw["entity_types"]}


def entity_types() -> list:
    return load_benchmark_ontology_raw()["entity_types"]
