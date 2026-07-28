"""
Builds a minimal, valid DrawingDocument (models/drawing_document.py) + a run directory
backed by LocalFsArtifactStore for one benchmark sheet — no platform infra required
(Conversion_Layer_Plan.md §0).
"""
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from pnid_agent.models.drawing_document import (
    AntivirusScan,
    DrawingDocument,
    DrawingPage,
    DrawingSource,
    FileType,
    PageNormalization,
    Raster,
)
from pnid_agent.storage.local_fs import LocalFsArtifactStore


def build_single_page_document(
    *, doc_id: str, job_id: str, tenant_id: str, image_path: str,
    artifact_store: LocalFsArtifactStore, dpi: int = 150,
) -> DrawingDocument:
    """One page, sourced from a single raster image file on disk (a real P&ID sheet).
    `image_path` must already exist and be readable (a PNG/JPG).

    IMPORTANT: `Raster.uri` must be a path RELATIVE TO THE JOB'S ARTIFACT DIRECTORY, not an
    absolute filesystem path — LocalFsArtifactStore._abs() raises ValueError("escapes
    job_dir") on anything that resolves outside `{root}/{job_id}/` (discovered by running
    the real stage_06_run against a doc built with an absolute uri — RasterDecodeFailed).
    So the source image gets COPIED into the store at the real convention path
    (`stage-00/pages/p{i}.png`, per Agent_Pipeline_Facts.md §1) rather than referenced by
    its original location."""
    path = Path(image_path)
    with Image.open(path) as img:
        width_px, height_px = img.size
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    size_bytes = len(raw_bytes)
    file_type = FileType.PNG if path.suffix.lower() == ".png" else FileType.JPG

    page_index = 0
    ext = path.suffix.lower().lstrip(".") or "png"
    relative_uri = f"stage-00/pages/p{page_index}.{ext}"
    artifact_store.write_bytes(job_id, relative_uri, raw_bytes)

    source = DrawingSource(
        file_type=file_type,
        original_filename=path.name,
        uploaded_at=datetime.now(timezone.utc),
        file_id=f"e2e-{doc_id}",
        n_pages_original=1,
        antivirus_scan=AntivirusScan(
            status="clean", scanned_at=datetime.now(timezone.utc), scanner="e2e_bench-skip",
        ),
    )
    page = DrawingPage(
        page_index=page_index,
        raster=Raster(
            uri=relative_uri, width_px=width_px, height_px=height_px, dpi=dpi,
            sha256=sha256, size_bytes=size_bytes,
        ),
        normalization=PageNormalization(steps_applied=[]),
    )
    return DrawingDocument(doc_id=doc_id, source=source, pages=[page], tenant_id=tenant_id, job_id=job_id)


def build_artifact_store(root: str) -> LocalFsArtifactStore:
    return LocalFsArtifactStore(root=root)
