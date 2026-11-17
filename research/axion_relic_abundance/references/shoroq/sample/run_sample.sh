#!/usr/bin/env bash
# Stage 1 (optional): rebuild the polytope sample from the Kreuzer-Skarke HuggingFace mirror and check that it reproduces
# references/shoroq/data/polytope_manifest_h11_004_050.json. Network-bound (HTTP range reads, about 1 GB of scan checkpoints). The
# committed manifest is the input of stage 2; this stage only demonstrates where it comes from.
source "$(dirname "$0")/../../../engine/common.sh"
require_producer_toolchain
S="$WORK/sample"; mkdir -p "$S"
log "stage 1: ordered KS scan"
"$PYTHON" "$REPO/references/shoroq/sample/ks_ordered_scan.py" "$S/ks_scan" 1000 4 50 "$WORKERS" 4 > "$WORK/logs/01_ks_scan.log" 2>&1
log "stage 1: legacy selection rule (verified against the committed manifest)"
"$PYTHON" "$REPO/references/shoroq/sample/select_polytopes.py" "$S/ks_scan/ordered_candidates.json" \
    "$REPO/references/shoroq/data/polytope_manifest_h11_004_050.json" "$S/selection.json" 100 4 50 > "$WORK/logs/01_select.log" 2>&1
"$PYTHON" "$REPO/references/shoroq/sample/build_g2_manifest.py" "$S/selection.json" "$S/polytope_manifest_h11_004_050.json" 100 4 50 20260930 \
    --reference "$REPO/references/shoroq/data/polytope_manifest_h11_004_050.json" | tee "$WORK/logs/01_manifest.log"
log "stage 1 done"
