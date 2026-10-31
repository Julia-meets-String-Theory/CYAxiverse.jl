"""Shared paths and helpers for the relic-abundance stage (read-only with respect to every input).

All locations come from the run configuration written by 02_write_run_config.py. Its path is the environment variable
SHOROQ_RELIC_CONFIG (default: $WORK/relic/run_config.json). Nothing here is tied to a particular machine.
"""
import hashlib, json, os

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCHEMA = os.path.join(REPO, "configs", "relic_output_schema.json")
GSTAR_TABLE = os.path.join(REPO, "data", "standardmodel2018.dat")


def config_path():
    p = os.environ.get("SHOROQ_RELIC_CONFIG")
    if p:
        return p
    work = os.environ.get("WORK")
    if not work:
        raise SystemExit("set SHOROQ_RELIC_CONFIG (or WORK) to locate the relic run configuration")
    return os.path.join(work, "relic", "run_config.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_config():
    return json.load(open(config_path()))


def canonical_modes():
    """Path of the A1 canonical mode table recorded in the run configuration."""
    return load_config()["input"]["canonical_mode_table"]


def dump_json(obj, path):
    """Deterministic JSON (sorted keys are NOT used: section order is meaningful); floats as shortest repr. Serialised in
    memory first (a non-finite value raises before anything is written) and renamed into place: never a truncated file."""
    text = json.dumps(obj, indent=1, ensure_ascii=True, allow_nan=False) + "\n"
    tmp = f"{path}.writing"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def read_table(path, table):
    """Read an output table with the column types of configs/relic_output_schema.json (not inferred): %.17g writes an
    integral float without a decimal point, so a column whose values are all integral would otherwise be read as int64
    (and e.g. 10 ** int64(-29) raises). NA cells become NaN in float columns."""
    import pandas as pd
    spec = json.load(open(SCHEMA))["tables"][table]
    dt = {c["name"]: float for c in spec if c["type"] == "float"}
    T = pd.read_csv(path, dtype=dt, na_values=["NA"], keep_default_na=False, float_precision="round_trip")
    for c in spec:                             # booleans also when the table is empty (no value to infer from)
        if c["type"] == "bool" and T[c["name"]].dtype != bool:
            T[c["name"]] = T[c["name"]].astype(str).str.lower().eq("true")
    return T


def check_run_location(cfg, relic_dir):
    """Stop if this run configuration was written for another relic directory (WORK moved or copied): its absolute paths
    would silently point to the other directory's files."""
    want = os.path.realpath(cfg["run"]["production_output_dir"]); have = os.path.realpath(relic_dir)
    if want != have:
        raise SystemExit(f"run_config.json was written for {want}, but this relic directory is {have}: the WORK directory was moved or "
                         "copied; rerun the relic stage in the new location (outputs are not relocatable)")
