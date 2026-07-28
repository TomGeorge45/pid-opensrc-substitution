"""Stage 11 deterministic ID allocators.

``temp_id`` and ``relation_id`` are page-scoped and deterministic — same
input → identical IDs across runs.

``temp_id`` links relations to entities INTERNALLY. The user-visible
identifier is the rive-sb UUID populated in ``entity.id`` when the
Tag-ID lookup matches an existing entity.
"""
from __future__ import annotations


def alloc_temp_id(page_index: int, position: int) -> str:
    """Return ``f'p{page_index}_e{position:04d}'``.

    Use the position of the entity in the page-sorted list (sorted by
    source detection_id) — guarantees stability across runs.
    """
    if page_index < 0:
        raise ValueError(f"page_index must be ≥0, got {page_index}")
    if position < 0:
        raise ValueError(f"position must be ≥0, got {position}")
    return f"p{page_index}_e{position:04d}"


def alloc_relation_id(page_index: int, position: int) -> str:
    """Return ``f'p{page_index}_r{position:04d}'``."""
    if page_index < 0:
        raise ValueError(f"page_index must be ≥0, got {page_index}")
    if position < 0:
        raise ValueError(f"position must be ≥0, got {position}")
    return f"p{page_index}_r{position:04d}"


def alloc_document_temp_id(page_index: int) -> str:
    """Return ``f'p{page_index}_doc'`` — the page's title-block document
    entity's temp ID. Distinct from the ``e{NNNN}`` form so it cannot
    collide with detection-derived entities.
    """
    if page_index < 0:
        raise ValueError(f"page_index must be ≥0, got {page_index}")
    return f"p{page_index}_doc"
