"""Stage 6 (Line Tracing) data model — vendored verbatim from pnid-intelligence-agent
(agents/pnid-intelligence-agent/pnid_agent/models/line_tracing.py), which is safely
portable per project_agent_history memory (intelligence-agent scrapped, no ownership
sensitivity). Only change from the source: no other change, kept byte-identical on
purpose so scoring code can rely on the same field semantics as the real Stage 6.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


StrokeStyle = Literal[
    "continuous",
    "dashed",
    "dotted",
    "double_line",
    "unknown",
]

LineType = Literal[
    "process",
    "electric_signal",
    "pneumatic",
    "capillary",
    "hydraulic",
    "unknown",
]

JunctionKind = Literal[
    "cross",
    "tee",
    "jumper",
    "joint",
]

EndpointKind = Literal["junction", "symbol", "loose_end"]


class Endpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: EndpointKind
    ref: Optional[str] = Field(default=None)
    position: Tuple[int, int]


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(pattern=r"^p\d+_g\d{4,}$")
    page_index: int = Field(ge=0)
    polyline: List[Tuple[int, int]] = Field(min_length=2)
    stroke_style: StrokeStyle
    line_type: LineType
    endpoint_a: Endpoint
    endpoint_b: Endpoint
    passes_through_symbols: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class Junction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    junction_id: str = Field(pattern=r"^p\d+_j\d{4,}$")
    page_index: int = Field(ge=0)
    position: Tuple[int, int]
    kind: JunctionKind
    ambiguous: bool
    incident_segment_ids: List[str] = Field(default_factory=list)


class PageLineGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_index: int = Field(ge=0)
    segments: List[Segment] = Field(default_factory=list)
    junctions: List[Junction] = Field(default_factory=list)
    stats: Dict[str, int] = Field(default_factory=dict)
