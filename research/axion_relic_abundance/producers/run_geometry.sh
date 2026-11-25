#!/usr/bin/env bash
# Producer, stage 2: G2 geometry generation (v1 recipe of the Shoroq sample: fast sampler, adaptive moduli policy, QCD window
# [25, 40], seed per resolve_seed.py: G2_SEED > manifest base_seed > 20260930) on the polytope manifest POLYTOPE_MANIFEST (required; the Shoroq sample is
# references/shoroq/data/polytope_manifest_h11_004_050.json, layout in DATA_CONTRACTS.md), then the repository materializer, the independent geometry audit, the Julia
# read-back cross-check, the physical spectra (prec 500, quartics, threshold H0) and the acceptance gate.
# Expensive: full sample measured 2026-10-29 with WORKERS = 4: about 1 h wall (G2 30 min, spectra 10 min). Override the h11 range
# with H11_MIN/H11_MAX and the polytopes per h11 with N_PER_H11 (smaller values are for tests only).
source "$(dirname "$0")/../engine/common.sh"
require_producer_toolchain
[[ -n "${POLYTOPE_MANIFEST:-}" ]] || die "set POLYTOPE_MANIFEST to the polytope manifest of the sample (see DATA_CONTRACTS.md)"
MANIFEST="$POLYTOPE_MANIFEST"
require "$MANIFEST" "$CYAX_REPO/scripts/build_orientifold_axion_database.py"
# Range and size default to the manifest (H11_MIN, H11_MAX, N_PER_H11 override them; the Shoroq manifest gives 4, 50, 100).
read -r MAN_MIN MAN_MAX MAN_N < <("$PYTHON" "$REPO/producers/geometry/resolve_seed.py" defaults "$MANIFEST") || die "cannot read the polytope manifest $MANIFEST"
H11_MIN="${H11_MIN:-$MAN_MIN}"; H11_MAX="${H11_MAX:-$MAN_MAX}"; N_PER_H11="${N_PER_H11:-$MAN_N}"
# Seed: G2_SEED (explicit) > manifest base_seed (or the base implied by its per-entry seeds) > 20260930 (Shoroq recipe v1); see
# producers/geometry/resolve_seed.py. The effective seed and its source are recorded in $WORK/provenance/seed_resolution.json.
mkdir -p "$WORK/provenance"
SEED=$("$PYTHON" "$REPO/producers/geometry/resolve_seed.py" resolve "$MANIFEST" "$WORK/provenance/seed_resolution.json" ${G2_SEED:+--cli-seed "$G2_SEED"}) || exit 1
[[ -e "$WORK/geometry" ]] && die "$WORK/geometry exists: refusing to overwrite a previous run"
RAW="$WORK/geometry/run_h11_$(printf %03d $H11_MIN)_$(printf %03d $H11_MAX)"
V="$WORK/validation"; mkdir -p "$V"/{materialization,julia,spectra,status,integrity} "$WORK/provenance"
G2=("$PYTHON" "$CYAX_REPO/scripts/generate_geometric_data_multitriangulation.py" --h11_min "$H11_MIN" --h11_max "$H11_MAX" --n "$N_PER_H11"
    --sampling-scheme fast --moduli-policy adaptive --qcd-volume-min 25 --qcd-volume-max 40 --seed "$SEED" --cores "$WORKERS"
    --database-source manifest --polytope-manifest "$MANIFEST" --verbose --outdir "$RAW")
"$PYTHON" "$REPO/producers/geometry/capture_provenance.py" "$WORK/provenance/geometry_provenance.json" --repo "$CYAX_REPO" \
    --julia-project "$CYAX_REPO" --julia "$JULIA_VMM" --manifest "$MANIFEST" --command "${G2[*]}"
log "stage 2: G2 generation"
"${G2[@]}" > "$WORK/logs/02_g2_generation.log" 2>&1
"$PYTHON" "$REPO/producers/geometry/status_ledgers.py" "$V/status" "$RAW" > "$WORK/logs/02_status.log"
"$PYTHON" "$REPO/producers/geometry/resolve_seed.py" verify "$WORK/provenance/seed_resolution.json" "$V/status/frst_proposal_status.csv" | tee -a "$WORK/logs/02_status.log"
log "stage 2: materialize for Julia"
"$PYTHON" "$REPO/producers/geometry/materialize_cyax_for_julia.py" --repo "$CYAX_REPO" "$RAW" "$WORK/materialized" \
    > "$V/materialization/materialize.jsonl" 2> "$WORK/logs/02_materialize.err"
log "stage 2: independent geometry audit"
AUD="$V/materialization/audit_with_materialized_h11_$(printf %03d $H11_MIN)_$(printf %03d $H11_MAX).jsonl"
"$PYTHON" "$REPO/producers/geometry/audit_geometry.py" "$CYAX_REPO/scripts" "$RAW" "$AUD" "$WORK/materialized" --workers "$WORKERS" \
    > "$WORK/logs/02_audit.log" 2>&1
"$PYTHON" "$REPO/producers/geometry/flag_qcd_edge.py" "$AUD" "$V/qcd_edge_flags.json" > "$WORK/logs/02_qcd_flags.log"
log "stage 2: Julia read-back cross-check"
CYAXIVERSE_DATA_DIR="$WORK/materialized" jvmm "$REPO/producers/geometry/julia_crosscheck.jl" "$V/julia/julia_crosscheck.jsonl" > "$WORK/logs/02_julia_crosscheck.log" 2>&1
"$PYTHON" "$REPO/producers/geometry/compare_julia.py" "$CYAX_REPO/scripts" "$RAW" "$WORK/materialized" "$V/julia/julia_crosscheck.jsonl" \
    "$V/julia/julia_compare.jsonl" > "$WORK/logs/02_julia_compare.log" 2>&1
"$PYTHON" "$REPO/producers/geometry/qkq_scale_aware.py" "$WORK/materialized" "$V/julia/julia_crosscheck.jsonl" "$V/julia/qkq_scale_aware.json" > /dev/null
log "stage 2: physical spectra (upstream batch_physical_spectrum.jl, prec 500, quartics, default threshold log10 H0)"
[[ -e "$WORK/spectra" ]] && { echo "$WORK/spectra exists: refusing to overwrite"; exit 1; }
cp -R "$WORK/materialized" "$WORK/spectra"
for h in $(ls "$WORK/spectra" | sed -n 's/^h11_0*//p'); do
  jvmm "$CYAX_REPO/scripts/batch_physical_spectrum.jl" --data-dir "$WORK/spectra" --h11 "$h" --prec 500 --quartics \
      --summary "$V/spectra/spectra_summary_h$h.csv" >> "$WORK/logs/02_spectra_h$h.log" 2>&1 || echo "batch exit nonzero h11=$h" >> "$WORK/logs/02_spectra_h$h.log" &
  while (( $(jobs -rp | wc -l) >= WORKERS )); do sleep 1; done
done
wait
log "stage 2: acceptance gate"
NGEO=$(find "$RAW" -name cyax.h5 | wc -l | tr -d ' ')
"$PYTHON" "$REPO/producers/geometry/validate_stage.py" "$V/gate.json" --audit "$AUD" --materialize "$V/materialization/materialize.jsonl" \
    --julia "$V/julia/julia_compare.jsonl" --spectra-csv "$V/spectra/spectra_summary_h*.csv" --spectra-log "$WORK/logs/02_spectra_h*.log" \
    --expected-geometries "$NGEO" > "$WORK/logs/02_gate.log" 2>&1 || true
"$PYTHON" "$REPO/producers/geometry/final_gate.py" "$V/gate.json" "$V/julia/qkq_scale_aware.json" "$V/qcd_edge_flags.json" "$AUD" "$V/final_gate.json"
log "stage 2: integrity sample (exact-rational chain + independent spectrum recompute)"
"$PYTHON" "$REPO/producers/geometry/integrity_sample.py" "$WORK" "$V/integrity/integrity_sample.json" > "$WORK/logs/02_integrity.log" 2>&1
CYAXIVERSE_DATA_DIR="$WORK/materialized" jvmm "$REPO/producers/geometry/integrity_spectra.jl" "$V/integrity/integrity_sample_files.txt" \
    "$WORK/spectra" "$V/integrity/integrity_spectra.tsv" >> "$WORK/logs/02_integrity.log" 2>&1
log "stage 2 done: $NGEO geometries"
