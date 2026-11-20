#!/usr/bin/env bash
# Create (if needed) and push this repo to Hugging Face Space
#   https://huggingface.co/spaces/waleedarif1011/essencenews
#
# Idempotent: safe to re-run. Requires a Hugging Face write token with Spaces access.
#
# Usage:
#   export HF_TOKEN=hf_...          # or: huggingface-cli login
#   ./scripts/deploy_hf_space.sh
#
# Env overrides:
#   HF_USER   (default: waleedarif1011)
#   HF_SPACE  (default: essencenews)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_USER="${HF_USER:-waleedarif1011}"
export HF_SPACE="${HF_SPACE:-essencenews}"
SPACE_ID="${HF_USER}/${HF_SPACE}"
SPACE_URL="https://huggingface.co/spaces/${SPACE_ID}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: missing command '$1'" >&2
    exit 1
  }
}

need_cmd git
need_cmd python3
need_cmd curl

if ! python3 -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "Installing huggingface_hub..."
  python3 -m pip install -U "huggingface_hub[cli]"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  # Fall back to token cached by `huggingface-cli login` / HF_* env already set
  TOKEN_FROM_CACHE="$(
    python3 - <<'PY'
from huggingface_hub import get_token
print(get_token() or "")
PY
  )"
  if [[ -z "$TOKEN_FROM_CACHE" ]]; then
    cat >&2 <<EOF
error: no HF_TOKEN and not logged in via huggingface-cli.

Create a write token (Spaces access) at:
  https://huggingface.co/settings/tokens

Then either:
  export HF_TOKEN=hf_...
  ./scripts/deploy_hf_space.sh

or:
  huggingface-cli login
  ./scripts/deploy_hf_space.sh
EOF
    exit 1
  fi
  export HF_TOKEN="$TOKEN_FROM_CACHE"
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "==> Ensuring Space exists: ${SPACE_ID}"
python3 - <<'PY'
import os

from huggingface_hub import HfApi

space_id = f"{os.environ['HF_USER']}/{os.environ['HF_SPACE']}"
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)

try:
    info = api.repo_info(repo_id=space_id, repo_type="space")
    print(f"    Space already exists ({info.id}).")
except Exception:
    url = api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    print(f"    Created Space: {url}")
PY

echo "==> Uploading Space runtime files"
python3 - <<'PY'
import os
from pathlib import Path

from huggingface_hub import HfApi, upload_folder

root = Path.cwd()
space_id = f"{os.environ['HF_USER']}/{os.environ['HF_SPACE']}"
token = os.environ["HF_TOKEN"]

# Keep the Space slim: app + model + sample dataset + README/assets.
# Skip notebooks, tests, docs, CI, download helpers, and large raw dumps.
ignore = [
    ".git*",
    ".github/",
    "docs/",
    "tests/",
    "scripts/",
    "notebooks/",
    "dataset/News_Category_Dataset_v3.json",
    "etl_output.csv",
    "structured_output.jsonl",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    ".ipynb_checkpoints/",
]

api = HfApi(token=token)
api.repo_info(repo_id=space_id, repo_type="space")

upload_folder(
    repo_id=space_id,
    repo_type="space",
    folder_path=str(root),
    path_in_repo=".",
    ignore_patterns=ignore,
    token=token,
    commit_message="Deploy EssenceNews Gradio Space",
)
print(f"Uploaded to https://huggingface.co/spaces/{space_id}")
PY

echo "==> Checking Space HTTP status"
sleep 3
HTTP_CODE="$(curl -s -o /dev/null -w '%{http_code}' "${SPACE_URL}" || true)"
echo "    GET ${SPACE_URL} → HTTP ${HTTP_CODE}"

if [[ "$HTTP_CODE" == "200" ]]; then
  echo "SUCCESS: Space is live at ${SPACE_URL}"
  echo "Confirm README Live Demo badge → ${SPACE_URL}"
  exit 0
fi

cat <<EOF
Space files pushed. Build may still be running (HTTP ${HTTP_CODE}).
Build logs: ${SPACE_URL}
Once the Space returns HTTP 200, the README Live Demo badge URL is ready
(expected slug: ${SPACE_ID}).
EOF
