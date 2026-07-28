#!/usr/bin/env bash
set -euo pipefail
#
# CPU-prep step for the extraction-agent local benchmark
# (notebooks/e2e_harness/ExtractionAgent_Local_GPUOnly.ipynb).
# Run this LOCALLY on the Mac (NOT in Colab), once per code change you want reflected
# in the next Colab run.
#
# Sibling script to scripts/package_agent_src_for_colab.sh (that one packages
# pnid-intelligence-agent for the ArmL/e2e_harness work; this one packages
# pnid-EXTRACTION-agent for Extraction_Agent_Local_Plan.md). Kept as a SEPARATE script,
# not a flag on the existing one, because the two agents' dependency trees, HF repo
# target, and included ground-truth data are all different — merging them would make
# either script harder to read for no benefit. Does NOT modify
# package_agent_src_for_colab.sh.
#
# What it does:
#   1. Zips the PRIVATE monorepo source the harness needs at runtime:
#        agents/pnid-extraction-agent  (the WHOLE agent — pnid_pipeline/, config.yaml,
#          requirements.txt, AND scripts/eval/ — this is where review_reads/<stem>/
#          reviewed_truth.json ground truth and score.py's revR-scoring functions live;
#          both are load-bearing for scoring in Colab, not just reference material)
#      together with THIS repo's own e2e_bench + extraction_local (src/e2e_bench,
#      src/extraction_local — untracked in pid-ml's git per `git status --short src/`,
#      so the public GitHub mirror of pid-ml does NOT have them; they must travel in
#      this same zip). `extraction_local` imports from `e2e_bench.backends.
#      parse_json_common` / `e2e_bench.backends.parse_molmo` / `e2e_bench.types`
#      (see extraction_local/qwen_call_llm.py, molmo_points.py's import lines) — the
#      WHOLE e2e_bench package travels together rather than cherry-picking individual
#      files, since it's small and other extraction_local modules may grow more
#      e2e_bench imports later without this script needing to change again.
#   2. Pushes the zip to a PRIVATE Hugging Face dataset repo.
#
# ⚠️ EXPLICITLY OUT OF SCOPE FOR THIS SCRIPT (do not add without separate sign-off):
# the 13 real sheet PDFs (AG_PNID / RIVE_LTTS_Sample trees) are marked Restricted/EAR99
# and are NOT included here. They need Tom's explicit, separate OK before ANY upload
# to HF (even private) — see Extraction_Agent_Local_Plan.md §10, user checkpoint 1, and
# the placeholder cell in ExtractionAgent_Local_GPUOnly.ipynb for exactly where/how
# they'd need to land in Colab once that sign-off happens. The 13 stems + their local
# paths are already enumerated in src/e2e_harness/score_revR_real_sheets.py's `SHEETS`
# list — reuse that, don't re-derive it.
#
# Per this project's standing rule (MEMORY.md: "No Google Drive, ever" / GPU-CPU split
# feedback): HF is the only approved shared-storage channel for Colab.
#
# IMPORTANT — before running, confirm with Tom that pushing this private company source
# to a Hugging Face dataset repo (even set private=True) is something he wants done from
# his own HF account. This script does NOT run itself; nothing was pushed automatically
# by the agent that wrote it. This script is WRITTEN, NOT EXECUTED, as part of the task
# that produced it — do not run it without that explicit go-ahead.
#
# Usage:
#   export HF_TOKEN=hf_...                                    # write access to your HF account
#   export HF_EXTRACTION_AGENT_SRC_REPO=timthy45/pnid-extraction-agent-src   # optional override
#   source /Users/tomgeorge/pid-ml/.venv-e2e/bin/activate        # has huggingface_hub already
#   bash /Users/tomgeorge/pid-ml/scripts/package_extraction_agent_src_for_colab.sh

AGENT_REPO="/Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform"
PID_ML="/Users/tomgeorge/pid-ml"
HF_REPO_ID="${HF_EXTRACTION_AGENT_SRC_REPO:-timthy45/pnid-extraction-agent-src}"
: "${HF_TOKEN:?Set HF_TOKEN in your shell first (export HF_TOKEN=hf_...)}"

for p in \
    "$AGENT_REPO/agents/pnid-extraction-agent" \
    "$AGENT_REPO/agents/pnid-extraction-agent/scripts/eval" \
    "$PID_ML/src/e2e_bench" \
    "$PID_ML/src/extraction_local"; do
  [ -d "$p" ] || { echo "MISSING: $p" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
mkdir -p "$STAGE/agents" "$STAGE/pid_ml_src"
cp -R "$AGENT_REPO/agents/pnid-extraction-agent" "$STAGE/agents/"
cp -R "$PID_ML/src/e2e_bench" "$STAGE/pid_ml_src/"
cp -R "$PID_ML/src/extraction_local" "$STAGE/pid_ml_src/"

# Sanity: the ground-truth + scorer this whole benchmark depends on must actually be
# in the staged copy before we zip it — fail loud here rather than silently shipping an
# empty eval directory to Colab.
[ -d "$STAGE/agents/pnid-extraction-agent/scripts/eval/review_reads" ] || {
  echo "MISSING after copy: scripts/eval/review_reads (reviewed_truth.json ground truth)" >&2
  exit 1
}
[ -f "$STAGE/agents/pnid-extraction-agent/scripts/eval/score.py" ] || {
  echo "MISSING after copy: scripts/eval/score.py (revR scoring functions)" >&2
  exit 1
}

# DO NOT include the 13 sheet PDFs — see the header comment above. Nothing in this
# staging step ever copies AG_PNID/RIVE_LTTS_Sample trees; this assertion exists purely
# as a tripwire in case a future edit to this script accidentally widens the copy list.
if find "$STAGE" -iname "*.pdf" | grep -q .; then
  echo "REFUSING TO PACKAGE: found PDF(s) under $STAGE — this script must never include" >&2
  echo "the 13 real sheet PDFs (Restricted/EAR99, needs separate explicit sign-off)." >&2
  find "$STAGE" -iname "*.pdf" >&2
  rm -rf "$STAGE"
  exit 1
fi

# Strip caches / vcs metadata / bytecode so the zip is small and doesn't drag in
# unrelated monorepo git history.
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name "*.pyc" -delete
find "$STAGE" -name ".git" -type d -prune -exec rm -rf {} + 2>/dev/null || true

GITSHA="$(git -C "$AGENT_REPO" rev-parse --short HEAD 2>/dev/null || echo nogit)"
ZIP_NAME="pnid_extraction_agent_src_${GITSHA}.zip"
( cd "$STAGE" && zip -qr "/tmp/$ZIP_NAME" . )
echo "built /tmp/$ZIP_NAME ($(du -h "/tmp/$ZIP_NAME" | cut -f1))"

python3 - "$HF_REPO_ID" "$ZIP_NAME" "$HF_TOKEN" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id, zip_name, token = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(
    path_or_fileobj=f"/tmp/{zip_name}",
    path_in_repo=f"agent_src/{zip_name}",
    repo_id=repo_id, repo_type="dataset", token=token,
)
# Stable pointer so the notebook config never has to change per push.
api.upload_file(
    path_or_fileobj=f"/tmp/{zip_name}",
    path_in_repo="agent_src/latest.zip",
    repo_id=repo_id, repo_type="dataset", token=token,
)
print(f"pushed agent_src/{zip_name} and agent_src/latest.zip to {repo_id} (private=True)")
PY

rm -rf "$STAGE"
echo "done. Set EXTRACTION_AGENT_SRC_REPO=\"$HF_REPO_ID\" in the notebook's config cell."
echo
echo "REMINDER: the 13 sheet PDFs are NOT in this zip (see header). They need a SEPARATE"
echo "explicit sign-off + separate upload before the notebook's Phase A/B cells can run."
