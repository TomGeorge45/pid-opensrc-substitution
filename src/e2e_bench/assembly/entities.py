"""
DetectionRecord -> EntityMapping -> build_entity -> BundleEntity (Conversion_Layer_Plan.md
§5, "assembly needed between stage 4 and the graph stages").

**Entities with no bbox are dropped** by build_entity/BundleEntity — every DetectionRecord
must carry provenance.bbox (it always does, per stage04_detection.py's use of the real
compose function) before reaching this step.
"""
from pnid_agent.models.detections import DetectionRecord
from pnid_agent.models.ontology_mapping import EntityMapping
from pnid_agent.models.rive_ontology import BundleEntity
from pnid_agent.stages.graph_construction.entities import build_entity

from ..ontology import entity_type_names


def detections_to_entities(
    *, detections: list, page_index: int, page_size: tuple, stage_4_model_version: str,
) -> tuple:
    """detections: list[DetectionRecord] (e.g. Stage04Output.pages[i].detections).
    Returns (entities: List[BundleEntity], detection_to_temp_id: Dict[str,str],
    entity_type_by_temp_id: Dict[str,str])."""
    names = entity_type_names()
    entities = []
    detection_to_temp_id = {}
    entity_type_by_temp_id = {}

    for i, det in enumerate(detections):
        temp_id = f"p{page_index}_e{i:04d}"
        mapping = EntityMapping(
            mapping_id=f"p{page_index}_m{i:03d}",
            source_detection_id=det.detection_id,
            page_index=page_index,
            entity_type=det.entity_type,
            entity_type_name=names.get(det.entity_type, det.entity_type),
            match_confidence=det.provenance.confidence,
        )
        entity, metadata, unresolved = build_entity(
            mapping, det, temp_id=temp_id, page_size=page_size,
            stage_4_model_version=stage_4_model_version,
        )
        if entity is None:
            continue  # dropped (e.g. no bbox) - should not happen given our detections always have one
        entities.append(entity)
        detection_to_temp_id[det.detection_id] = temp_id
        entity_type_by_temp_id[temp_id] = det.entity_type

    return entities, detection_to_temp_id, entity_type_by_temp_id
