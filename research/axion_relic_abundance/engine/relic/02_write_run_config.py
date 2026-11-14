"""Write the run configuration of one relic-abundance run from the committed template.

The template (configs/relic_production_config.json) holds every approved constant, rule and expected count of the frozen
production run. This script only fills the run-specific locations and checksums: the relic-input directory and manifest,
the A1 canonical tables, the g* table and the output directory. No number in the template is changed (checked below).
Usage: python 02_write_run_config.py TEMPLATE RELIC_INPUT_DIR A1_DIR OUTPUT_DIR OUT_CONFIG
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GSTAR_TABLE, dump_json, sha256

template, input_dir, a1_dir, output_dir, out = sys.argv[1:6]
cfg = json.load(open(template))
a1 = json.load(open(os.path.join(a1_dir, "a1_input_verification.json")))
if not a1["all_checks_pass"]:
    sys.exit("A1 checks did not pass; no run configuration written")
input_dir = os.path.abspath(input_dir)
modes = os.path.abspath(os.path.join(a1_dir, "canonical_modes.csv"))
if sha256(modes) != a1["outputs"]["canonical_modes.csv"]:
    sys.exit("canonical_modes.csv changed after A1")
if sha256(os.path.join(a1_dir, "canonical_geometries.csv")) != a1["outputs"]["canonical_geometries.csv"]:
    sys.exit("canonical_geometries.csv changed after A1")
if sha256(GSTAR_TABLE) != cfg["cosmology"]["gstar_table_sha256"]:
    sys.exit("g* table checksum differs from the approved table")

I = cfg["input"]
I["input_dir"] = input_dir
I["manifest"] = os.path.join(input_dir, "relic_abundance_input_manifest.json")
I["manifest_sha256"] = a1["manifest_sha256"]
I["input_sha256sums_file"] = os.path.join(input_dir, "SHA256SUMS.txt")
I["input_sha256sums_file_sha256"] = a1["input_sha256sums_file_sha256"]
I["spectra_view"] = os.path.join(input_dir, "spectra")
I["canonical_mode_table"] = modes
I["canonical_mode_table_sha256"] = a1["outputs"]["canonical_modes.csv"]
I["canonical_geometry_table"] = os.path.abspath(os.path.join(a1_dir, "canonical_geometries.csv"))
I["canonical_geometry_table_sha256"] = a1["outputs"]["canonical_geometries.csv"]   # the relic driver takes the geometry population from it
I["spectrum_set_sha256"] = a1["spectrum_set_sha256"]
cfg["run"]["production_output_dir"] = os.path.abspath(output_dir)
cfg["cosmology"]["gstar_table"] = GSTAR_TABLE


def leaves(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{pre}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{pre}[{i}]")
    else:
        yield pre, x


t = dict(leaves(json.load(open(template)))); c = dict(leaves(cfg))
changed = [k for k, v in t.items() if isinstance(v, (bool, int, float)) and c[k] != v]
if changed:
    sys.exit(f"numeric template values changed: {changed}")
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
dump_json(cfg, out)
print("wrote", out)
