#!/usr/bin/env bash
# Stage 5: final small-angle analysis (0 < theta <= 0.5, no lower cut): exact TIGHT intervals, N_TIGHT(theta) maxima,
# robustness (smoothing, bootstrap, tight-edge sensitivity), h11 dependence, for the primary population (theta/) and the labelled
# sensitivity population including partially valid geometries (theta_sensitivity_partial_validity/); then an independent Python
# cross-check of both.
source "$(dirname "$0")/common.sh"
F="$WORK/final_analysis"; S="$REPO/engine/theta"
require "$WORK/relic/run_config.json" "$WORK/relic/tables/run_metadata.json" "$WORK/relic/validation/production_verification.json"
grep -q '"all_pass": true' "$WORK/relic/validation/production_verification.json" || die "stage 4 verification did not pass"
[[ -e "$F" ]] && die "$F exists: refusing to overwrite a previous run"
[[ -n "${RELIC_RUN:-}" ]] || die "set RELIC_RUN to the run file of the relic run (theta_analysis overrides)"
# Retry safety: a stage that fails part-way must not leave a final_analysis/ that blocks the documented rerun. The partial output is
# kept for diagnosis under a new name (never deleted); rerunning this stage then starts clean.
STAGE5_OK=0
stage5_cleanup() {
    if [[ "$STAGE5_OK" != 1 && -e "$F" ]]; then
        local k="$F.failed.$(date -u +%Y%m%dT%H%M%SZ)"; mv "$F" "$k" && echo "stage 5 failed: partial output kept in $k; fix the cause and rerun this stage" >&2
    fi
}
trap stage5_cleanup EXIT
# The scenarios of stage 5 are those of the relic run (relic/run_config.json). A different run file would only change the theta_analysis overrides,
# so a run file whose scenarios differ from the relic run's is refused instead of being ignored silently.
"$PYTHON" - "$RELIC_RUN" "$WORK/relic/run_config.json" <<'PY' || die "RELIC_RUN differs from the run file of the relic stage (scenario labels, T_reh or w_R): use the same run file for stages 4 and 5"
import json, sys
run, cfg = (json.load(open(p)) for p in sys.argv[1:3])
mv = cfg.get("methodology_version", 1)
key = lambda s: (s["label"], float(s["T_reh_GeV"]), float(s.get("w_R", cfg["reheating"].get("w_R_default", 0.0))) if mv == 2 else 0.0)
want = [key(s) for s in cfg["reheating"]["scenarios"]]
have = [key(s) if mv == 1 or "w_R" in s else (s["label"], float(s["T_reh_GeV"]), 0.0) for s in run["scenarios"]]
sys.exit(0 if want == have else 1)
PY
log "stage 5: theta configuration from the v1 theta methodology${RELIC_EXPECTATIONS:+ and the expectations}"
"$PYTHON" "$REPO/engine/relic/00_compose_run_config.py" theta "$REPO/configs/theta_methodology_v1.json" "$RELIC_RUN" "$F/theta_config.json" \
    ${RELIC_EXPECTATIONS:+--expectations "$RELIC_EXPECTATIONS"} | tee "$WORK/logs/05_compose.log"
export SHOROQ_THETA_CONFIG="$F/theta_config.json"
log "stage 5: verify relic run and inputs"
jana "$S/00_verify_inputs.jl" > "$WORK/logs/05_verify.log"
log "stage 5: theta analysis"
jana "$S/01_theta_final.jl" > "$WORK/logs/05_theta.log"
# independent cross-check of both populations: primary (theta/) and the labelled sensitivity population (theta_sensitivity_partial_validity/)
THETA_POPULATION=PRIMARY "$PYTHON" "$S/02_crosscheck_theta.py" > "$WORK/logs/05_crosscheck.log"
grep -q "ALL CHECKS PASS: True" "$WORK/logs/05_crosscheck.log" || { echo "theta cross-check (primary) failed"; exit 1; }
THETA_POPULATION=SENSITIVITY "$PYTHON" "$S/02_crosscheck_theta.py" > "$WORK/logs/05_crosscheck_sensitivity.log"
grep -q "ALL CHECKS PASS: True" "$WORK/logs/05_crosscheck_sensitivity.log" || { echo "theta cross-check (sensitivity) failed"; exit 1; }
log "stage 5 done"
STAGE5_OK=1
