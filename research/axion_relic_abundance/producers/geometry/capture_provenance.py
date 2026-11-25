"""Record the provenance G2 does NOT store per artifact (repository commits, package/solver versions, platform,
input-manifest hash, exact command lines), to be kept next to a generated sample.
Usage: python capture_provenance.py OUT_JSON --repo PATH --julia-project PATH --julia BIN --manifest PATH [--command "..."]...
"""
import argparse, datetime, hashlib, importlib.metadata, json, platform, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--repo", required=True); ap.add_argument("--julia-project", required=True)
ap.add_argument("--julia", required=True); ap.add_argument("--manifest", required=True)
ap.add_argument("--command", action="append", default=[])
a = ap.parse_args()
git = lambda repo, *args: subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()
ver = {}
for pkg in ("cytools", "numpy", "scipy", "h5py", "qpsolvers", "cvxopt", "osqp", "highspy", "ortools", "python-flint", "mosek", "pplpy", "pyarrow"):
    try:
        ver[pkg] = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        ver[pkg] = None
import qpsolvers  # noqa: E402
from cytools import config  # noqa: E402
rec = {
    "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "platform": {"system": platform.platform(), "machine": platform.machine(), "python": sys.version},
    "python_packages": ver, "qpsolvers_available": sorted(qpsolvers.available_solvers),
    "projection_solver_order_used_by_G2": (["mosek"] if config.mosek_is_activated() else []) + sorted(s for s in qpsolvers.available_solvers if s != "mosek"),
    "mosek_activated": bool(config.mosek_is_activated()),
    "tip_solver_rule": "CYTools tip_of_stretched_cone(1): MOSEK if activated; else OSQP for h11<25, HiGHS LP (approximate tip) for h11>=25",
    "generator_repo": {"path": a.repo, "commit": git(a.repo, "rev-parse", "HEAD"), "dirty": bool(git(a.repo, "status", "--porcelain")),
                       "remote_vmm": git(a.repo, "rev-parse", "origin/vmm")},
    "julia_project": {"path": a.julia_project, "commit": git(a.julia_project, "rev-parse", "HEAD"),
                      "dirty_src_or_scripts": bool(git(a.julia_project, "status", "--porcelain", "--", "src", "scripts"))},
    "julia_version": subprocess.run([a.julia, "--version"], capture_output=True, text=True).stdout.strip(),
    "polytope_manifest": {"path": a.manifest, "sha256": hashlib.sha256(open(a.manifest, "rb").read()).hexdigest()},
    "commands": a.command,
}
json.dump(rec, open(a.out, "w"), indent=1)
print(json.dumps(rec, indent=1))
