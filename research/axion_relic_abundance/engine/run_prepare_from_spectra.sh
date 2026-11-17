#!/usr/bin/env bash
# Relic input from existing CYAxiverse physical spectra (alternative to stages 2-3; no geometry or spectrum is computed).
# Entries with vacuum_validation.state = "VERIFY" are checked by the theta = 0 inertia scan on the potential stored in their
# file (CYAxiverse vmm environment, about 1 s per geometry); see engine/input/build_input_from_spectra.py and
# DATA_CONTRACTS.md. Writes WORK/relic_input (then run run_04_relic.sh and, optionally, run_05_theta.sh).
# Usage: engine/run_prepare_from_spectra.sh SPECTRA_LIST
source "$(dirname "$0")/common.sh"
[[ $# -eq 1 ]] || die "usage: $0 SPECTRA_LIST (schema cyaxiverse_relic_spectra/1)"
LIST="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; IN="$WORK/relic_input"; V="$WORK/vacuum_check"
require "$LIST"
[[ -e "$IN" || -e "$V" ]] && die "$IN or $V exists: refusing to overwrite a previous run"
mkdir -p "$V/data"
log "stage 0: spectra to verify by the theta = 0 inertia scan"
"$PYTHON" - "$LIST" "$V/data" "$V/verify_list.tsv" <<'PY'
import json, os, re, sys
L = json.load(open(sys.argv[1])); base = os.path.dirname(sys.argv[1]); rows = []
for e in L["spectra"]:
    if (e.get("vacuum_validation") or {}).get("state") != "VERIFY":
        continue
    p = e["path"] if os.path.isabs(e["path"]) else os.path.normpath(os.path.join(base, e["path"]))
    gid = e.get("geometry_id") or "/".join(p.split("/")[-4:-1])
    m = re.fullmatch(r"h11_(\d{3})/np_(\d{7})/cy_(\d{7})", gid)
    if not m:
        sys.exit(f"geometry_id {gid!r} must be h11_XXX/np_XXXXXXX/cy_XXXXXXX")
    link = os.path.join(sys.argv[2], gid, "cyax.h5"); os.makedirs(os.path.dirname(link), exist_ok=True); os.symlink(p, link)
    rows.append("\t".join(str(int(x)) for x in m.groups()))
open(sys.argv[3], "w").write("h11\tposition\tfrst\n" + "".join(r + "\n" for r in rows))
print(len(rows), "geometries to verify")
PY
SCAN=()
if (( $(wc -l < "$V/verify_list.tsv") > 1 )); then
  require_producer_toolchain             # VERIFY entries need CYAxiverse (inertia scan); DOCUMENTED-only input does not
  log "stage 0: theta = 0 inertia scan"
  CYAXIVERSE_DATA_DIR="$V/data" jvmm "$REPO/producers/vacuum/00_theta0_inertia_scan.jl" "$V/verify_list.tsv" "$V/data" "$V/inertia_scan.jsonl" \
      > "$WORK/logs/00_inertia_scan.log" 2>&1
  SCAN=(--inertia-scan "$V/inertia_scan.jsonl")
fi
log "stage 0: relic input"
"$PYTHON" "$REPO/engine/input/build_input_from_spectra.py" "$LIST" "$IN" ${SCAN[@]+"${SCAN[@]}"} | tee "$WORK/logs/00_build_input.log"
log "stage 0 done"
