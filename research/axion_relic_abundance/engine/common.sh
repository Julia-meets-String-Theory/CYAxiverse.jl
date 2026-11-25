# Sourced by every run_*.sh and regression test: loads configs/paths.env, checks the toolchain and defines helpers.
# Not executable on its own.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVFILE="${SHOROQ_PATHS:-$REPO/configs/paths.env}"
die() { echo "ERROR: $*" >&2; exit 1; }
[[ -f "$ENVFILE" ]] || die "missing $ENVFILE (copy configs/paths.env.example and edit it, or set SHOROQ_PATHS)"
_WORK_FROM_ENV="${WORK:-}"
source "$ENVFILE"
[[ -n "$_WORK_FROM_ENV" ]] && WORK="$_WORK_FROM_ENV"      # an exported WORK overrides paths.env (used by the regression tests)
# engine toolchain (every stage): Python and Julia 1.8.4
for v in PYTHON JULIA_ANALYSIS WORK; do [[ -n "${!v:-}" ]] || die "$v is not set in $ENVFILE"; done
[[ -x "$PYTHON" ]] || die "PYTHON=$PYTHON is not executable"
[[ -x "$JULIA_ANALYSIS" ]] || die "JULIA_ANALYSIS=$JULIA_ANALYSIS is not executable"
# producer toolchain (geometry/spectrum/vacuum producers and the theta = 0 inertia scan of VERIFY entries only): CYAxiverse.jl
# checkout at the pinned revision and Julia 1.12.7. The dataset was produced with CYAxiverse.jl vmm a5eacd8; another revision
# may change geometries or spectra.
CYAX_PINNED=a5eacd8cd4a5905161cab239a461ba252c64e8e0
# Inside a CYAxiverse.jl checkout (research/axion_relic_abundance/) an unset CYAX_REPO defaults to that checkout.
if [[ -z "${CYAX_REPO:-}" && -f "$REPO/../../scripts/batch_physical_spectrum.jl" ]]; then CYAX_REPO="$(cd "$REPO/../.." && pwd)"; fi
require_producer_toolchain() {
  for v in CYAX_REPO JULIA_VMM; do [[ -n "${!v:-}" ]] || die "$v is not set in $ENVFILE (needed by the producers and by VERIFY entries)"; done
  [[ -x "$JULIA_VMM" ]] || die "JULIA_VMM=$JULIA_VMM is not executable"
  [[ -f "$CYAX_REPO/scripts/generate_geometric_data_multitriangulation.py" && -f "$CYAX_REPO/scripts/batch_physical_spectrum.jl" ]] \
      || die "CYAX_REPO=$CYAX_REPO is not a CYAxiverse.jl checkout with the vmm scripts"
  CYAX_HEAD="$(git -C "$CYAX_REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ "$CYAX_HEAD" != "$CYAX_PINNED" ]]; then
    [[ "${ALLOW_OTHER_CYAX_COMMIT:-0}" == 1 ]] || die "CYAX_REPO is at $CYAX_HEAD, not the pinned $CYAX_PINNED (set ALLOW_OTHER_CYAX_COMMIT=1 to run anyway; the results must then be re-validated)"
    echo "WARNING: CYAX_REPO at $CYAX_HEAD, not the pinned $CYAX_PINNED" >&2
  fi
}
WORKERS="${WORKERS:-8}"
export WORK
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0            # git status must not create index.lock in read-only checkouts
# Julia without startup file: the CYAxiverse project for stages 2-3, the pinned analysis project for stages 4-5.
jvmm() { if [[ -n "${JULIA_VMM_DEPOT:-}" ]]; then JULIA_DEPOT_PATH="$JULIA_VMM_DEPOT" "$JULIA_VMM" --startup-file=no --project="$CYAX_REPO" "$@"; else "$JULIA_VMM" --startup-file=no --project="$CYAX_REPO" "$@"; fi; }
jana() { if [[ -n "${JULIA_ANALYSIS_DEPOT:-}" ]]; then JULIA_DEPOT_PATH="$JULIA_ANALYSIS_DEPOT" "$JULIA_ANALYSIS" --startup-file=no --project="$REPO/environment/julia_analysis" "$@"; else "$JULIA_ANALYSIS" --startup-file=no --project="$REPO/environment/julia_analysis" "$@"; fi; }
# require PATH...: stop with a clear message when an input file or directory is missing
require() { local p; for p in "$@"; do [[ -e "$p" ]] || die "required input missing: $p"; done; }
mkdir -p "$WORK/logs"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$WORK/logs/pipeline.log"; }
