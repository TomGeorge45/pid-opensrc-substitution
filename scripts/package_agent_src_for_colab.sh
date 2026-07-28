#!/usr/bin/env bash
set -euo pipefail
#
# CPU-prep step for Arm L (notebooks/e2e_harness/ArmL_QwenVL_FullStack_GPUOnly.ipynb).
# Run this LOCALLY on the Mac (NOT in Colab), once per code change you want reflected
# in the next Colab run.
#
# What it does:
#   1. Zips the PRIVATE monorepo source the harness needs at runtime:
#        agents/pnid-intelligence-agent, shared/entity_operations, shared/rive_adk,
#        shared/security  (from rive-ai-platform, editable-installed in .venv-e2e today)
#      together with THIS repo's own e2e_bench/e2e_harness (src/e2e_bench, src/e2e_harness
#      — these are untracked in pid-ml's git, per `git status --short src/`, so the public
#      GitHub mirror of pid-ml does NOT have them; they must travel in this same zip).
#   2. Pushes the zip to a PRIVATE Hugging Face dataset repo.
#
# Per this project's standing rule (MEMORY.md: "No Google Drive, ever" / GPU-CPU split
# feedback): HF is the only approved shared-storage channel for Colab, so this is HF,
# not Drive, and this packaging step is the "CPU prep, local" half of the split — the
# Colab notebook only ever downloads + installs, never builds this zip itself.
#
# IMPORTANT — before running, confirm with Tom that pushing this private company source
# to a Hugging Face dataset repo (even set private=True) is something he wants done from
# his own HF account. This script does NOT run itself; nothing was pushed automatically
# by the agent that wrote it.
#
# Usage:
#   export HF_TOKEN=hf_...                              # write access to your HF account
#   export HF_AGENT_SRC_REPO=timthy45/pnid-agent-src     # optional override, must be one
#                                                         # you can create/write as private
#   source /Users/tomgeorge/pid-ml/.venv-e2e/bin/activate   # has huggingface_hub already
#   bash /Users/tomgeorge/pid-ml/scripts/package_agent_src_for_colab.sh

AGENT_REPO="/Users/tomgeorge/Developer/work/Rive-Platform/rive-ai-platform"
PID_ML="/Users/tomgeorge/pid-ml"
HF_REPO_ID="${HF_AGENT_SRC_REPO:-timthy45/pnid-agent-src}"
: "${HF_TOKEN:?Set HF_TOKEN in your shell first (export HF_TOKEN=hf_...)}"

for p in \
    "$AGENT_REPO/agents/pnid-intelligence-agent" \
    "$AGENT_REPO/shared/entity_operations" \
    "$AGENT_REPO/shared/rive_adk" \
    "$AGENT_REPO/shared/security" \
    "$PID_ML/src/e2e_bench" \
    "$PID_ML/src/e2e_harness"; do
  [ -d "$p" ] || { echo "MISSING: $p" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
mkdir -p "$STAGE/agents" "$STAGE/shared" "$STAGE/pid_ml_src"
cp -R "$AGENT_REPO/agents/pnid-intelligence-agent" "$STAGE/agents/"
cp -R "$AGENT_REPO/shared/entity_operations" "$STAGE/shared/"
cp -R "$AGENT_REPO/shared/rive_adk" "$STAGE/shared/"
cp -R "$AGENT_REPO/shared/security" "$STAGE/shared/"
cp -R "$PID_ML/src/e2e_bench" "$STAGE/pid_ml_src/"
cp -R "$PID_ML/src/e2e_harness" "$STAGE/pid_ml_src/"

# Strip caches / vcs metadata / bytecode so the zip is small and doesn't drag in
# unrelated monorepo git history.
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name "*.pyc" -delete
find "$STAGE" -name ".git" -type d -prune -exec rm -rf {} + 2>/dev/null || true

GITSHA="$(git -C "$AGENT_REPO" rev-parse --short HEAD 2>/dev/null || echo nogit)"
ZIP_NAME="pnid_agent_src_${GITSHA}.zip"
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
echo "done. Set AGENT_SRC_REPO=\"$HF_REPO_ID\" in the Arm L notebook's config cell."
