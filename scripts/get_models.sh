#!/usr/bin/env bash
set -euo pipefail

# Exact acquisition of the two weight repositories this submission runs on.
#
# On the competition machine both are already on the shared /model volume, and every
# measurement shipped here was taken from those paths:
#   MODEL=/model/ModelScope/Qwen/Qwen3.8-27B-FP8
#   DRAFT_DFLASH=/model/ModelScope/incoai/Qwen3.8-27B-DFlash2
#
# Off that machine, download the same two ModelScope repositories and point the run
# script at the local copies:
#   MODEL=/root/infer/models/Qwen3.8-27B-FP8 \
#   DRAFT_DFLASH=/root/infer/models/Qwen3.8-27B-DFlash2 bash scripts/run_sglang_48g.sh
#
# The DSpark comparison draft is fetched and sha256-verified by setup_sglang_48g.sh.

DEST=${DEST:-/root/infer/models}
HERE=$(cd "$(dirname "$0")" && pwd)
# The commits this submission pins; scripts/models.sha256 lists their files.
MAIN_REV=416b6c4c4ff599f76304db98ae7b5518973df13c
DRAFT_REV=5cfe71fd56ed83d5ca27f04c1345cc7d862bbe4f

python3 -m pip show modelscope >/dev/null 2>&1 || pip install -q modelscope

# The Python API, because the `modelscope download` flags differ between releases.
python3 - "$DEST" "$MAIN_REV" "$DRAFT_REV" <<'PY'
import sys

from modelscope import snapshot_download

dest, main_rev, draft_rev = sys.argv[1:]
snapshot_download("Qwen/Qwen3.8-27B-FP8", revision=main_rev, local_dir=f"{dest}/Qwen3.8-27B-FP8", max_workers=4)
snapshot_download("incoai/Qwen3.8-27B-DFlash2", revision=draft_rev, local_dir=f"{dest}/Qwen3.8-27B-DFlash2", max_workers=4)
PY

(cd "$DEST" && sha256sum --quiet -c "$HERE/models.sha256")
echo "weights match scripts/models.sha256"
