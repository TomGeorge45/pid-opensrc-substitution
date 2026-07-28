import tempfile
from types import SimpleNamespace

import pytest

from e2e_bench.assembly.document import build_artifact_store, build_single_page_document

# A real P&ID crop, already used throughout this project's benchmarking work.
REAL_IMAGE_PATH = (
    "/private/tmp/claude-501/-Users-tomgeorge-pid-ml/a852824c-b8e0-473e-bcfc-7bdbd11a58f6"
    "/scratchpad/skid_example_unmarked.png"
)


@pytest.fixture
def store():
    return build_artifact_store(tempfile.mkdtemp())


@pytest.fixture
def doc(store):
    return build_single_page_document(
        doc_id="sheet-001", job_id="job-001", tenant_id="benchmark",
        image_path=REAL_IMAGE_PATH, artifact_store=store,
    )


@pytest.fixture
def context():
    return SimpleNamespace(tenant_id="benchmark")
