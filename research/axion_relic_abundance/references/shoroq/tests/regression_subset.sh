#!/usr/bin/env bash
# Regression of stages 2-3 on a subset (default: the first 12 polytopes at h11 = 11, which include the theta = 0 saddle
# h11_011/np_0000012 repaired at a validated minimum). The real stage drivers are run; every geometric, potential and
# spectral dataset, every vacuum-repair evidence record and every relic-input manifest field must equal the frozen ones.
# Cost: about 10 minutes, mostly Julia compilation.
# Usage: references/shoroq/tests/regression_subset.sh FROZEN_ROOT [WORK] [H11] [N]
#   FROZEN_ROOT  copy of the frozen reference, layout documented in references/shoroq/tests/verify_frozen_reference.py
#   WORK         output directory (default: a new temporary directory); summary in WORK/regression_subset.json
#   H11, N       subset: positions 1..N at one h11 (positions keep their seeds 20260930 + position)
source "$(dirname "$0")/../../../engine/common.sh"
require_producer_toolchain
[[ $# -ge 1 ]] || die "usage: $0 FROZEN_ROOT [WORK] [H11] [N]"
[[ -d "$1" ]] || die "FROZEN_ROOT $1 is not a directory (see references/shoroq/tests/verify_frozen_reference.py for the expected layout)"
FR="$(cd "$1" && pwd)"; WORK="${2:-$(mktemp -d)}"; H="${3:-11}"; N="${4:-12}"; export WORK; mkdir -p "$WORK/logs"
[[ -e "$WORK/geometry" ]] && die "$WORK/geometry exists: use an empty WORK directory"
FU="$FR/thesis_g2_adaptive/unified_h11_004_050"; FACC="$FR/spectrum_count_audit/results/accounting_all.jsonl"
FRES="$FR/tachyonic_vacua_audit/results"; FCOR="$FR/tachyonic_vacua_audit/corrected_spectra"; FMAN="$FR/relic_abundance_input/relic_abundance_input_manifest.json"
require "$FU/geometry" "$FU/materialized" "$FU/spectra" "$FACC" "$FRES" "$FCOR" "$FMAN"
L="$WORK/logs"; RES="$WORK/regression_subset.json"
exec 3>&1
NAMES=(); STATUS=()
record() { NAMES+=("$1"); STATUS+=("$2"); echo "$2  $1" >&3; }
run() { local name="$1"; shift; if "$@"; then record "$name" PASS; else record "$name" FAIL; fi; }

run "frozen reference intact (checksums)" "$PYTHON" "$REPO/references/shoroq/tests/verify_frozen_reference.py" "$FR" "$WORK/frozen_reference_subset.json" subset > "$L/frozen_reference.log"
export H11_MIN="$H" H11_MAX="$H" N_PER_H11="$N" POLYTOPE_MANIFEST="$WORK/subset_manifest.json"
"$PYTHON" - "$REPO/references/shoroq/data/polytope_manifest_h11_004_050.json" "$POLYTOPE_MANIFEST" "$H" "$N" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); h, n = int(sys.argv[3]), int(sys.argv[4])
d["polytopes"] = [p for p in d["polytopes"] if p["h11"] == h and p["unified_position"] <= n]
json.dump(d, open(sys.argv[2], "w"))
PY
run "stage 2 driver (G2, materialize, audit, Julia read-back, spectra, gates, integrity)" "$REPO/producers/run_geometry.sh" > "$L/stage2_driver.log" 2>&1
run "stage 2 final gate PASS" grep -q '"overall": "PASS"' "$WORK/validation/final_gate.json"
run "stage 2 integrity sample: exact-rational chain and independent spectrum recompute" \
    "$PYTHON" "$REPO/references/shoroq/tests/check_integrity_sample.py" "$WORK/validation/integrity" "$WORK/integrity_check.json"
run "stage 2 outputs identical to frozen (raw, materialized, spectra; accepted set)" \
    "$PYTHON" "$REPO/references/shoroq/tests/compare_geometry_subset.py" "$WORK" "$FU" "$WORK/compare_geometry.json" > "$L/compare_geometry.log"
EXP=$("$PYTHON" - "$FACC" "$WORK/validation/status/polytope_status.csv" <<'PY'
import csv, json, sys
acc = {(int(r["h11"]), int(r["polytope_index"])) for r in csv.DictReader(open(sys.argv[2])) if r["status"] == "accepted"}
print(sum(1 for l in open(sys.argv[1]) if (lambda r: (r["h11"], r["position"]) in acc and r["inertia_below_minus_H0"] > 0)(json.loads(l))))
PY
)
run "stage 3 driver (inertia scan, minima, validation, corrected spectra, relic input; $EXP tachyonic expected)" \
    env EXPECTED_TACHYONIC="$EXP" "$REPO/producers/run_vacuum_repair.sh" > "$L/stage3_driver.log" 2>&1
run "stage 3 outputs identical to frozen (evidence records, corrected spectra, classification, manifest records)" \
    "$PYTHON" "$REPO/references/shoroq/tests/compare_vacuum_repair.py" "$WORK" "$FACC" "$FRES" "$FCOR" "$FMAN" "$WORK/compare_vacuum_repair.json" > "$L/compare_vacuum_repair.log"
run "QCD-edge flag rule reproduces the frozen 539 flags on all 4491 audited geometries" \
    "$PYTHON" "$REPO/references/shoroq/tests/check_qcd_flags_frozen.py" "$FR/thesis_g2_adaptive" "$WORK/qcd_flags_check" > "$L/qcd_flags.log"

"$PYTHON" - "$RES" "${#NAMES[@]}" "${NAMES[@]}" "${STATUS[@]}" <<'PY'
import json, sys
out, n = sys.argv[1], int(sys.argv[2]); names, st = sys.argv[3:3 + n], sys.argv[3 + n:3 + 2 * n]
res = {"test": "regression_subset", "checks": [{"check": a, "status": b} for a, b in zip(names, st)],
       "n_pass": st.count("PASS"), "n_fail": st.count("FAIL")}
res["verdict"] = "PASS" if res["n_fail"] == 0 else "FAIL"
json.dump(res, open(out, "w"), indent=1)
print(f"REGRESSION {res['verdict']}: {res['n_pass']} passed, {res['n_fail']} failed -> {out}")
sys.exit(0 if res["n_fail"] == 0 else 1)
PY
