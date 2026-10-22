#!/usr/bin/env bash
# Engine-only launcher; requires a complete checkpoint and packed trunk.
# 24 threads is the conservative lab01 kernel-test choice, not a full-model speed claim.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?usage: run-lab01.sh <model-dir> <trunk-dir> [k3 options]}"
TRUNK="${2:?usage: run-lab01.sh <model-dir> <trunk-dir> [k3 options]}"
shift 2
[ -d "$MODEL" ] && [ -d "$TRUNK" ] || { echo 'model/trunk directories must exist' >&2; exit 2; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"
exec "$ROOT/bin/k3" "$MODEL" --trunk "$TRUNK" --preset desktop --trunk-gb 24 --cache-gb 1 --incremental "$@"
