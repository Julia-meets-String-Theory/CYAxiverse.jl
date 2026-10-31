#!/usr/bin/env bash
# General-sample tests of the engine (no expected population count of any reference case is used; a few minutes).
# Inputs are frozen Shoroq spectra, read in place (never written); the frozen tables serve only as row-by-row references.
#   1. configuration composer: invalid scenarios, labels, theta settings and expectation overrides are refused
#   2. one geometry, direct spectra input (vacuum VERIFY by the theta = 0 inertia scan), the three Shoroq scenarios
#   3. subset (20 theta = 0 geometries over h11 4-40, 3 corrected spectra with DOCUMENTED_EXTERNAL evidence, one
#      theta = 0 tachyonic and one undocumented geometry excluded by policy), Shoroq scenarios + custom T_reh
#   2-3: every master, geometry-summary and canonical row of a Shoroq scenario equals its frozen row; independent
#        verification C (every row) and B (50 digits, stratified) pass; theta analysis and its cross-check pass
#   4. direct-input refusals: unusable vacuum under policy "refuse", changed evidence file checksum
#   5. the frozen reference directory is not modified
# Usage: tests/test_general.sh FROZEN_ROOT WORK
source "$(dirname "$0")/../engine/common.sh"
require_producer_toolchain
[[ $# -eq 2 ]] || die "usage: $0 FROZEN_ROOT WORK"
FR="$(cd "$1" && pwd)"; T="$2"; mkdir -p "$T"; T="$(cd "$T" && pwd)"
[[ -e "$T/one" || -e "$T/subset" ]] && die "$T is not empty"
H="$REPO/tests/general_helpers.py"; CMP="$REPO/engine/relic/00_compose_run_config.py"; SH="$REPO/references/shoroq/configs"
NAMES=(); STATUS=()
exec 3>&1
record() { NAMES+=("$1"); STATUS+=("$2"); echo "$2  $1" >&3; }
run() { local name="$1"; shift; if "$@"; then record "$name" PASS; else record "$name" FAIL; fi; }
refused() { local name="$1"; shift; if "$@" > "$T/last_refusal.log" 2>&1; then record "$name" FAIL; else record "$name (refused: $(tail -1 "$T/last_refusal.log" | cut -c1-90))" PASS; fi; }
date -u +%FT%TZ > "$T/start_marker"
mkdir -p "$T/configs"

# 1. composer
runfile() { echo "{\"schema\": \"cyaxiverse_relic_run/1\", \"scenarios\": $1${2:+, \"theta_analysis\": $2}}" > "$T/configs/$3.json"; }
runfile '[{"label": "low", "T_reh_GeV": 1e-9}]' "" below_Teq
runfile '[{"label": "high", "T_reh_GeV": 1e17}]' "" above_table
runfile '[{"label": "a", "T_reh_GeV": 1e4}, {"label": "a", "T_reh_GeV": 1e5}]' "" duplicate
runfile '[{"label": "bad/label", "T_reh_GeV": 1e4}]' "" badlabel
runfile '[{"label": "x", "T_reh_GeV": 1e4}]' '{"theta": {"theta_max": 1.5}}' thetamax
runfile '[{"label": "x", "T_reh_GeV": 1e4}]' '{"theta": {"theta_min": 1e-5}}' thetamin
runfile '[{"label": "bbn", "T_reh_GeV": 0.003}]' "" bbn
echo '{"relic": {"cosmology": {"H0_eV": 1.0}}}' > "$T/configs/bad_expectations.json"
refused "composer: T_reh below the g* table (T_eq = 8e-10 GeV lies below it)" "$PYTHON" "$CMP" relic "$REPO/configs/methodology_v1.json" "$T/configs/below_Teq.json" "$T/configs/x.json"
refused "composer: T_reh above the g* table" "$PYTHON" "$CMP" relic "$REPO/configs/methodology_v1.json" "$T/configs/above_table.json" "$T/configs/x.json"
refused "composer: duplicate scenario label" "$PYTHON" "$CMP" relic "$REPO/configs/methodology_v1.json" "$T/configs/duplicate.json" "$T/configs/x.json"
refused "composer: label unusable in file names" "$PYTHON" "$CMP" relic "$REPO/configs/methodology_v1.json" "$T/configs/badlabel.json" "$T/configs/x.json"
refused "composer: theta_max > 1" "$PYTHON" "$CMP" theta "$REPO/configs/theta_methodology_v1.json" "$T/configs/thetamax.json" "$T/configs/x.json"
refused "composer: lower theta cutoff" "$PYTHON" "$CMP" theta "$REPO/configs/theta_methodology_v1.json" "$T/configs/thetamin.json" "$T/configs/x.json"
refused "composer: expectations cannot change the methodology" "$PYTHON" "$CMP" relic "$REPO/configs/methodology_v1.json" "$SH/run.json" "$T/configs/x.json" --expectations "$T/configs/bad_expectations.json"
run "composer: T_reh = 3 MeV accepted with a BBN warning recorded" bash -c \
    "'$PYTHON' '$CMP' relic '$REPO/configs/methodology_v1.json' '$T/configs/bbn.json' '$T/configs/bbn_template.json' --checks '$T/configs/bbn_checks.json' > /dev/null && grep -q BBN '$T/configs/bbn_checks.json'"

# 2-3. samples
cat > "$T/configs/custom.json" <<'EOF'
{"schema": "cyaxiverse_relic_run/1", "description": "Shoroq scenarios plus custom reheating temperatures",
 "scenarios": [{"label": "5MeV", "T_reh_GeV": 0.005}, {"label": "1e4GeV", "T_reh_GeV": 1e4}, {"label": "1e10GeV", "T_reh_GeV": 1e10},
               {"label": "20MeV", "T_reh_GeV": 0.02}, {"label": "1e7GeV", "T_reh_GeV": 1e7}, {"label": "1e14GeV", "T_reh_GeV": 1e14}],
 "theta_analysis": {"optimisation": {"bootstrap_resamples": 200}}}
EOF
sample() {   # sample KIND RUNFILE
  local K="$1" W="$T/$1"
  run "$K: spectra list from frozen spectra" "$PYTHON" "$H" make-list "$FR" "$T/configs/${K}_list.json" "$K" > /dev/null
  run "$K: direct spectra input (inertia-scan VERIFY, documented evidence, exclusions by policy)" \
      env WORK="$W" "$REPO/engine/run_prepare_from_spectra.sh" "$T/configs/${K}_list.json" > "$T/${K}_prepare.log" 2>&1
  run "$K: relic stage (A1 + relic + schema validation + independent C/B verification)" env WORK="$W" RELIC_RUN="$2" "$REPO/engine/run_relic.sh" > "$T/${K}_relic.log" 2>&1
  run "$K: every Shoroq-scenario row identical to the frozen row" "$PYTHON" "$H" compare-rows "$W" "$FR" "$T/${K}_rows.json" > "$T/${K}_rows.log"
  run "$K: theta stage (input checks, analysis, Python cross-check)" env WORK="$W" RELIC_RUN="$2" "$REPO/engine/run_theta.sh" > "$T/${K}_theta.log" 2>&1
}
sample one "$SH/run.json"
sample subset "$T/configs/custom.json"
run "subset: vacuum states as designed (20 inertia-verified, 3 documented, 1 tachyonic and 1 unverified excluded)" "$PYTHON" - "$T/subset/relic_input/relic_abundance_input_manifest.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
print(s)
sys.exit(0 if s["by_vacuum_validation"] == {"VERIFIED_THETA0_INERTIA": 20, "DOCUMENTED_EXTERNAL": 3, "TACHYONIC_AT_THETA0": 1, "UNVERIFIED": 1}
         and s["excluded_geometries"] == 2 else 1)
PY

# 4. refusals of the direct input
"$PYTHON" "$H" make-list "$FR" "$T/configs/refuse_list.json" subset refuse > /dev/null
refused "direct input: unusable vacuum stops the build under policy refuse" env WORK="$T/refuse" "$REPO/engine/run_prepare_from_spectra.sh" "$T/configs/refuse_list.json"
"$PYTHON" "$H" tamper-evidence "$T/configs/subset_list.json" "$T/configs/tampered_list.json"
refused "direct input: evidence file checksum mismatch" "$PYTHON" "$REPO/engine/input/build_input_from_spectra.py" "$T/configs/tampered_list.json" "$T/tampered_input"

# 5. frozen data untouched
find "$FR" -newer "$T/start_marker" \( -type f -o -type l -o -type d \) > "$T/frozen_modified.txt" 2>/dev/null || true
run "frozen reference directory not modified" test ! -s "$T/frozen_modified.txt"

"$PYTHON" - "$T/test_general.json" "${#NAMES[@]}" "${NAMES[@]}" "${STATUS[@]}" <<'PY'
import json, sys
out, n = sys.argv[1], int(sys.argv[2]); names, st = sys.argv[3:3 + n], sys.argv[3 + n:3 + 2 * n]
res = {"test": "test_general", "checks": [{"check": a, "status": b} for a, b in zip(names, st)], "n_pass": st.count("PASS"), "n_fail": st.count("FAIL")}
res["verdict"] = "PASS" if res["n_fail"] == 0 else "FAIL"
json.dump(res, open(out, "w"), indent=1)
print(f"GENERAL TESTS {res['verdict']}: {res['n_pass']} passed, {res['n_fail']} failed -> {out}")
sys.exit(0 if res["n_fail"] == 0 else 1)
PY
