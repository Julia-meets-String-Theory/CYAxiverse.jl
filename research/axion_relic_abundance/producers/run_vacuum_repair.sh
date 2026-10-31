#!/usr/bin/env bash
# Stage 3: theta = 0 inertia scan of every accepted geometry, identification of the geometries whose theta = 0
# production mass matrix has tachyonic directions with |m| >= H0, their leading-branch minima (repository
# leading_critical_branches), validation of each minimum on the FULL potential at ~500 and ~1000 digits, the physical
# spectrum about the validated minimum (production engine and writer), and the relic-input manifest.
# Cost: inertia scan about 0.85 CPU-hours for 4491 geometries (run in WORKERS chunks); the repair itself a few minutes.
source "$(dirname "$0")/../engine/common.sh"
require_producer_toolchain
[[ -n "${POLYTOPE_MANIFEST:-}" ]] || die "set POLYTOPE_MANIFEST to the polytope manifest used by producers/run_geometry.sh"
V="$WORK/vacuum_repair"; S="$REPO/producers/vacuum"
require "$WORK/materialized" "$WORK/spectra" "$WORK/validation/status/polytope_status.csv" "$WORK/validation/qcd_edge_flags.json" "$WORK/validation/final_gate.json"
grep -q '"overall": "PASS"' "$WORK/validation/final_gate.json" || die "stage 2 final gate did not pass ($WORK/validation/final_gate.json)"
[[ -e "$V" ]] && die "$V exists: refusing to overwrite a previous run"
mkdir -p "$V/chunks"
export CYAXIVERSE_DATA_DIR="$WORK/materialized"
log "stage 3: theta = 0 inertia scan"
"$PYTHON" - "$WORK/validation/status/polytope_status.csv" "$V/accepted_list.tsv" <<'PY'
import csv, sys
rows = sorted((int(r["h11"]), int(r["polytope_index"])) for r in csv.DictReader(open(sys.argv[1])) if r["status"] == "accepted")
open(sys.argv[2], "w").write("h11\tposition\n" + "".join(f"{h}\t{p}\n" for h, p in rows))
PY
"$PYTHON" - "$V/accepted_list.tsv" "$V/chunks" "$WORKERS" <<'PY'
import sys
lines = open(sys.argv[1]).read().splitlines(); hdr, rows = lines[0], lines[1:]; n = int(sys.argv[3])
for k in range(n):
    open(f"{sys.argv[2]}/chunk_{k}.tsv", "w").write("\n".join([hdr] + rows[k::n]) + "\n")
PY
for k in $(seq 0 $((WORKERS - 1))); do
  jvmm "$S/00_theta0_inertia_scan.jl" "$V/chunks/chunk_$k.tsv" "$WORK/spectra" "$V/chunks/scan_$k.jsonl" > "$WORK/logs/03_scan_$k.log" 2>&1 &
done
wait
"$PYTHON" - "$V/accepted_list.tsv" "$V/chunks" "$WORKERS" "$V/00_theta0_inertia_scan.jsonl" "$V/tachyonic_list.tsv" <<'PY'
import json, sys
order = [tuple(map(int, l.split("\t"))) for l in open(sys.argv[1]).read().splitlines()[1:]]
recs = {}
for k in range(int(sys.argv[3])):
    for l in open(f"{sys.argv[2]}/scan_{k}.jsonl"):
        r = json.loads(l); recs[(r["h11"], r["position"])] = l
assert len(recs) == len(order), "inertia scan incomplete"
bad = [json.loads(recs[k]) for k in order if "error" in json.loads(recs[k])]
assert not bad, f"inertia scan errors: {bad[:3]}"
open(sys.argv[4], "w").write("".join(recs[k] for k in order))
tach = [k for k in order if json.loads(recs[k])["inertia_below_minus_H0"] > 0]   # tachyonic directions with |m| >= H0
open(sys.argv[5], "w").write("h11\tposition\n" + "".join(f"{h}\t{p}\n" for h, p in tach))
print(len(order), "geometries scanned;", len(tach), "with theta = 0 tachyons |m| >= H0")
PY
NTACH=$(( $(wc -l < "$V/tachyonic_list.tsv") - 1 ))
if (( NTACH == 0 )); then
  # Nothing to repair (possible only for small test subsets): the repository vacua CLI refuses an empty selection, so
  # steps 01-04 and 06 are skipped and their evidence files are written empty; every geometry keeps its theta = 0 spectrum.
  log "stage 3: no theta = 0 tachyonic geometry: vacuum-repair steps 01-04 and 06 have nothing to do"
  for f in 01_theta0_structure 02_vacua_auto_groups 03_minima_validation 04_vacuum_spectra 06_verify_corrected; do : > "$V/$f.jsonl"; done
  "$PYTHON" "$S/05_build_relic_input.py" "$WORK" "$POLYTOPE_MANIFEST" --expected-tachyonic 0 | tee "$WORK/logs/03_05.log"
  "$PYTHON" "$S/07_verify_manifest.py" "$WORK/relic_input/relic_abundance_input_manifest.json" "$V/07_manifest_verification.json"
  log "stage 3 done (no repair needed)"
  exit 0
fi
log "stage 3: structure of the theta = 0 saddles and exact leading-branch minima"
jvmm "$S/01_identify_theta0.jl" "$V/tachyonic_list.tsv" "$WORK/spectra" "$V/01_theta0_structure.jsonl" > "$WORK/logs/03_01.log" 2>&1
log "stage 3: repository vacua CLI (--method auto) on writable copies"
mkdir -p "$V/vacua_cli/data"; GEOS=()
while IFS=$'\t' read -r h p; do
  [[ "$h" == "h11" ]] && continue
  g=$(printf "h11_%03d/np_%07d/cy_0000001" "$h" "$p"); mkdir -p "$V/vacua_cli/data/$(dirname "$g")"
  cp -R "$WORK/materialized/$g" "$V/vacua_cli/data/$g"; GEOS+=(--geometry "$h,$p,1")
done < "$V/tachyonic_list.tsv"
jvmm "$CYAX_REPO/scripts/batch_vacua_pipeline.jl" --data-dir "$V/vacua_cli/data" --method auto --summary "$V/vacua_cli/vacua_summary.csv" "${GEOS[@]}" \
    > "$WORK/logs/03_vacua_auto.log" 2>&1
jvmm "$S/02b_read_vacua_groups.jl" "$V/vacua_cli/data" "$V/tachyonic_list.tsv" "$V/02_vacua_auto_groups.jsonl" > "$WORK/logs/03_02b.log" 2>&1
log "stage 3: validate minima on the full potential (~500 and ~1000 digits)"
jvmm "$S/03_validate_minima.jl" "$V/tachyonic_list.tsv" "$V/03_minima_validation.jsonl" > "$WORK/logs/03_03.log" 2>&1
log "stage 3: physical spectra about the validated minima"
jvmm "$S/04_vacuum_spectra.jl" "$V/tachyonic_list.tsv" "$WORK/spectra" "$V/corrected_spectra" "$V/04_vacuum_spectra.jsonl" > "$WORK/logs/03_04.log" 2>&1
log "stage 3: relic-input manifest"
"$PYTHON" "$S/05_build_relic_input.py" "$WORK" "$POLYTOPE_MANIFEST" ${EXPECTED_TACHYONIC:+--expected-tachyonic $EXPECTED_TACHYONIC} \
    | tee "$WORK/logs/03_05.log"
"$PYTHON" - "$WORK/relic_input/relic_abundance_input_manifest.json" "$V/corrected_records.tsv" "$WORK" <<'PY'
import json, os, sys
M = json.load(open(sys.argv[1]))
rows = [r for r in M["geometries"] if r["status"] == "VACUUM_CORRECTED"]
open(sys.argv[2], "w").write("geometry_id\th11\tposition\tspectrum_source\n" +
    "".join(f"{r['geometry_id']}\t{r['h11']}\t{r['position']}\t{os.path.join(sys.argv[3], r['spectrum_source'])}\n" for r in rows))
PY
jvmm "$S/06_verify_corrected.jl" "$V/corrected_records.tsv" "$V/06_verify_corrected.jsonl" "$WORK/spectra" > "$WORK/logs/03_06.log" 2>&1
"$PYTHON" - "$V/06_verify_corrected.jsonl" <<'PY'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
assert r and all(x["pass"] for x in r), "end-to-end check of the corrected spectra failed"
print(len(r), "corrected spectra pass the end-to-end check")
PY
"$PYTHON" "$S/07_verify_manifest.py" "$WORK/relic_input/relic_abundance_input_manifest.json" "$V/07_manifest_verification.json"
log "stage 3 done"
