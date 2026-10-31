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
