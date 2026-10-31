#!/usr/bin/env bash
# Regression of stages 4-5 against the frozen accepted results (cheap: a few minutes).
# Runs this repository's relic and theta code on the FROZEN relic input and requires byte-identical scientific outputs;
# also reruns the pre-production benchmark (real + synthetic boundary modes) and applies every relic validator.
# Usage: references/shoroq/tests/regression_relic_theta.sh FROZEN_ROOT [WORK]
#   FROZEN_ROOT  copy of the frozen reference, layout documented in references/shoroq/tests/verify_frozen_reference.py
#   WORK         output directory (default: a new temporary directory); summary in WORK/regression_relic_theta.json
source "$(dirname "$0")/../../../engine/common.sh"
[[ $# -ge 1 ]] || die "usage: $0 FROZEN_ROOT [WORK]"
[[ -d "$1" ]] || die "FROZEN_ROOT $1 is not a directory (see references/shoroq/tests/verify_frozen_reference.py for the expected layout)"
FR="$(cd "$1" && pwd)"; WORK="${2:-$(mktemp -d)}"; export WORK; mkdir -p "$WORK/logs"
[[ -e "$WORK/relic" ]] && die "$WORK/relic exists: use an empty WORK directory"
IN="$FR/relic_abundance_input"; FT="$FR/relic_abundance_production/tables"; FTH="$FR/relic_abundance_final_analysis/results/theta"
FB="$FR/relic_abundance_production_prep/benchmarks"; FBS="$FR/relic_abundance_production_prep/results/benchmark_summary.json"
require "$IN/relic_abundance_input_manifest.json" "$FT" "$FTH" "$FB/benchmark_modes.csv" "$FB/out_A_julia" "$FBS"
RES="$WORK/regression_relic_theta.json"; S="$REPO/engine/relic"; R="$WORK/relic"; L="$WORK/logs"; mkdir -p "$R"
exec 3>&1
NAMES=(); STATUS=()
record() { NAMES+=("$1"); STATUS+=("$2"); echo "$2  $1" >&3; }
run() { local name="$1"; shift; if "$@"; then record "$name" PASS; else record "$name" FAIL; fi; }

run "frozen reference intact (checksums)" "$PYTHON" "$REPO/references/shoroq/tests/verify_frozen_reference.py" "$FR" "$WORK/frozen_reference_relic.json" relic > "$L/frozen_reference.log"
SH="$REPO/references/shoroq/configs"
run "relic template composed from methodology v1 + Shoroq run + expectations equals the frozen configuration" bash -c \
    "'$PYTHON' '$S/00_compose_run_config.py' relic '$REPO/configs/methodology_v1.json' '$SH/run.json' '$R/template.json' --expectations '$SH/expectations.json' --checks '$R/scenario_checks.json' > '$L/compose.log' && cmp -s '$R/template.json' '$SH/relic_production_config.frozen.json'"
run "A1 canonical tables built, all A1 checks pass" "$PYTHON" "$S/01_build_mode_table.py" "$R/template.json" "$IN" "$R/a1" \
    --expect-manifest-sha256 4a5d3270216d0748cfd49f3806f65963362fc8ba92a18c4f125cce6852ecc6a2 > "$L/a1.log"
run "A1 canonical_modes.csv byte-identical (sha256 662054...)" \
    test "$(shasum -a 256 "$R/a1/canonical_modes.csv" | cut -d' ' -f1)" = 662054232d15de063f6d97c847f04ce303b1c635d84771d8801e465ee130c4a7
run "A1 canonical_geometries.csv byte-identical" cmp -s "$R/a1/canonical_geometries.csv" "$FR/relic_abundance_production_prep/results/canonical_geometries.csv"
run "run configuration written (no template number changed)" "$PYTHON" "$S/02_write_run_config.py" "$R/template.json" "$IN" "$R/a1" "$R" "$R/run_config.json" > "$L/run_config.log"
run "relic run (Julia 1.8.4, analysis project)" jana "$S/compute_relic_abundance.jl" "$R/run_config.json" "$R/a1/canonical_modes.csv" "$R/tables" abundance > "$L/relic.log" 2>&1
for f in master_modes_{5MeV,1e4GeV,1e10GeV}.csv geometry_summary_{5MeV,1e4GeV,1e10GeV}.csv h11_summary_{5MeV,1e4GeV,1e10GeV}.csv global_summary.json; do
  run "relic/$f byte-identical" cmp -s "$FT/$f" "$R/tables/$f"
done
run "relic run_metadata: only script/config paths and checksums differ" "$PYTHON" - "$FT/run_metadata.json" "$R/tables/run_metadata.json" > "$L/run_metadata_diff.log" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
print("run_metadata keys differing:", diff)
sys.exit(0 if set(diff) <= {"script", "script_sha256", "config", "config_sha256", "mode_table"} else 1)
PY
export SHOROQ_RELIC_CONFIG="$R/run_config.json"
run "relic output schema validation" "$PYTHON" "$S/03_validate_outputs.py" "$R/tables" 16313 > "$L/validate.log"
run "relic independent verification (C on every row, B stratified, 50 digits)" "$PYTHON" "$S/04_verify_relic_outputs.py" "$R" > "$L/verify.log"
run "relic verification results identical to the frozen run" "$PYTHON" - "$FR/relic_abundance_production/validation/production_verification.json" "$R/validation/production_verification.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
sys.exit(0 if all(a[k] == b[k] for k in ("C_full", "B_sample", "B_geometries")) else 1)
PY
run "benchmark rerun (real + synthetic boundary modes)" jana "$S/compute_relic_abundance.jl" "$R/run_config.json" "$FB/benchmark_modes.csv" "$WORK/benchmark_rerun" abundance > "$L/benchmark.log" 2>&1
run "benchmark: rerun identical, production rows identical, A-vs-B within tolerance" \
    "$PYTHON" "$REPO/references/shoroq/tests/compare_relic_benchmark.py" "$WORK/benchmark_rerun" "$FB" "$FBS" "$R/tables" "$WORK/relic_benchmark_comparison.json" > "$L/benchmark_compare.log"
mkdir -p "$WORK/final_analysis"; export SHOROQ_THETA_CONFIG="$WORK/final_analysis/theta_config.json"
run "theta configuration composed from theta methodology v1 + Shoroq expectations equals the frozen configuration" bash -c \
    "'$PYTHON' '$S/00_compose_run_config.py' theta '$REPO/configs/theta_methodology_v1.json' '$SH/run.json' '$SHOROQ_THETA_CONFIG' --expectations '$SH/expectations.json' > '$L/compose_theta.log' && cmp -s '$SHOROQ_THETA_CONFIG' '$SH/theta_analysis_config.frozen.json'"
run "relic: abundance validity classes of the frozen population (4162 EVALUATED_POSITIVE, 329 PARTIAL_VALIDITY, none zero or all-excluded)" "$PYTHON" - "$R/tables" <<'PY'
import csv, sys
from collections import Counter
ok = all(Counter(r["abundance_validity"] for r in csv.DictReader(open(f"{sys.argv[1]}/geometry_abundance_validity_{l}.csv"))) ==
         Counter({"EVALUATED_POSITIVE": 4162, "PARTIAL_VALIDITY": 329}) for l in ("5MeV", "1e4GeV", "1e10GeV"))
sys.exit(0 if ok else 1)
PY
run "theta input verification" jana "$REPO/engine/theta/00_verify_inputs.jl" > "$L/theta_verify.log" 2>&1
run "theta analysis (primary and sensitivity populations)" jana "$REPO/engine/theta/01_theta_final.jl" > "$L/theta.log" 2>&1
# Approved 2026-10-31: the primary theta population excludes the 329 PARTIAL_VALIDITY geometries; the labelled sensitivity
# population (primary + PARTIAL_VALIDITY) is exactly the frozen population and must reproduce every frozen theta output:
# every CSV byte for byte, the summary JSON with identical content apart from its added "population" label.
TS="$WORK/final_analysis/theta_sensitivity_partial_validity"
for f in $(cd "$FTH" && find . -type f -name '*.csv' | sort); do
  run "sensitivity theta/${f#./} byte-identical to frozen" cmp -s "$FTH/$f" "$TS/$f"
done
run "sensitivity theta_summary_final.json identical to frozen (content; only the population label added)" "$PYTHON" - "$FTH/theta_summary_final.json" "$TS/theta_summary_final.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
lab = b.pop("population", None)
sys.exit(0 if a == b and lab == "SENSITIVITY_INCLUDING_PARTIAL_VALIDITY" else 1)
PY
run "theta independent cross-check, primary population (Python)" env THETA_POPULATION=PRIMARY "$PYTHON" "$REPO/engine/theta/02_crosscheck_theta.py" > "$L/crosscheck.log"
run "theta independent cross-check, sensitivity population (Python)" env THETA_POPULATION=SENSITIVITY "$PYTHON" "$REPO/engine/theta/02_crosscheck_theta.py" > "$L/crosscheck_sensitivity.log"
run "sensitivity cross-check results identical to the frozen run" "$PYTHON" - "$FR/relic_abundance_final_analysis/results/verification/crosscheck_python.json" "$WORK/final_analysis/verification/crosscheck_python_sensitivity_partial_validity.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
sys.exit(0 if a["theta"] == b["theta"] and a["new6"] == b["new6"] and all(v["agree"] for v in b["census"].values()) else 1)
PY
run "primary population: denominators 4162 (primary), 4491 (sensitivity), 0 not evaluated; N_max 25 / 40 / 29 (5MeV / 1e4GeV / 1e10GeV)" "$PYTHON" - "$WORK/final_analysis/theta/theta_population_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["scenarios"]
ok = all(s[l]["primary"]["n_geometries"] == 4162 and s[l]["sensitivity_partial_validity"]["n_geometries"] == 4491 and
         s[l]["excluded_not_evaluated"]["n_geometries"] == 0 and s[l]["primary"]["N_tight_max"] == n for l, n in (("5MeV", 25), ("1e4GeV", 40), ("1e10GeV", 29)))
sys.exit(0 if ok else 1)
PY

"$PYTHON" - "$RES" "${#NAMES[@]}" "${NAMES[@]}" "${STATUS[@]}" <<'PY'
import json, sys
out, n = sys.argv[1], int(sys.argv[2]); names, st = sys.argv[3:3 + n], sys.argv[3 + n:3 + 2 * n]
res = {"test": "regression_relic_theta", "checks": [{"check": a, "status": b} for a, b in zip(names, st)],
       "n_pass": st.count("PASS"), "n_fail": st.count("FAIL")}
res["verdict"] = "PASS" if res["n_fail"] == 0 else "FAIL"
json.dump(res, open(out, "w"), indent=1)
print(f"REGRESSION {res['verdict']}: {res['n_pass']} passed, {res['n_fail']} failed -> {out}")
sys.exit(0 if res["n_fail"] == 0 else 1)
PY
