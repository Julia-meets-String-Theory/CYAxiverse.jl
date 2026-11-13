"""Kinetic-normalization labels for the Python readers and writers (CYAX-0232).

Standard library only, so that light tools such as
``merge_orientifold_shards.py`` can apply the R-007 checks without importing
the CYTools stack. The label values and their meaning are fixed by the CYAX-0232
specification (R-006).
"""

from __future__ import annotations

from typing import Any, Iterable

KINETIC_NORMALIZATION = "sugra_c1"
KINETIC_NORMALIZATION_LABELS = (
    "sugra_c1",
    "p96_reproduction_c_half",
    "a96_reproduction_raw_radian",
    "legacy_c_half",
)

# F5: the reproduction JSON of reproduce_fuzzy_axions_h11_4.py.
F5_REPRODUCTION_SCHEMA = "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.3"
F5_LEGACY_REPRODUCTION_SCHEMA = "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.2"
# Inferred only for a file that matches a recorded layout (Clarification A2).
F5_A2_REPRODUCTION_SCHEMAS = {
    "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.0": "1.0",
    "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.1": "1.1",
}


def f5_legacy_layouts(version: str) -> list[Any]:
    """Return the recorded A2 layouts of F5 ``…-reproduction-<version>``.

    The layouts are recorded in ``f5_legacy_layouts_v10a.py``,
    ``f5_legacy_layouts_v10b.py`` and ``f5_legacy_layouts_v11.py`` (CYAX-0232
    T041); they are imported on first use.
    """

    from f5_legacy_layouts_v10a import LAYOUTS as v10a
    from f5_legacy_layouts_v10b import LAYOUTS as v10b
    from f5_legacy_layouts_v11 import LAYOUTS as v11

    return [layout for layout in (*v10a, *v10b, *v11) if layout.version == version]


def match_f5_legacy_layouts(summary: dict[str, Any], version: str) -> list[str]:
    """Return the ids of the recorded layouts that ``summary`` matches.

    A file matches a layout when R ⊆ (its normalized path set) ⊆ R ∪ O, with
    the file normalized against that layout's map nodes (Clarification A2).
    """

    from f5_legacy_layout_api import normalize

    paths_by_nodes: dict[Any, frozenset[str]] = {}
    matched = []
    for layout in f5_legacy_layouts(version):
        paths = paths_by_nodes.get(layout.map_nodes)
        if paths is None:
            paths = normalize(summary, layout.map_nodes)
            paths_by_nodes[layout.map_nodes] = paths
        if layout.required <= paths <= layout.required | layout.optional:
            matched.append(layout.id)
    return matched


def _unrecorded_paths(summary: dict[str, Any], version: str) -> list[str]:
    from f5_legacy_layout_api import normalize

    layouts = f5_legacy_layouts(version)
    nodes = frozenset().union(*(layout.map_nodes for layout in layouts))
    known = frozenset().union(*(layout.required | layout.optional for layout in layouts))
    return sorted(normalize(summary, nodes) - known)


def f5_reproduction_label(summary: dict[str, Any], source: Any) -> tuple[str, bool]:
    """Return ``(label, inferred)`` for an F5 reproduction JSON summary.

    - ``…-reproduction-1.3`` needs the top-level label ``"sugra_c1"`` (F5 is a
      primary physical family): ``("sugra_c1", False)``.
    - ``…-reproduction-1.2`` without a label: ``("legacy_c_half", True)``.
    - ``…-reproduction-1.0`` or ``…-1.1`` without a label that matches at
      least one recorded layout of its version (Clarification A2):
      ``("legacy_c_half", True)``. The label states the file's provenance; the
      only K-dependent fields of these layouts are the model-stage values.

    Raises a ``ValueError`` naming ``source`` for a missing or wrong label, a
    legacy version that carries a label, a 1.0 or 1.1 file that matches no
    recorded layout, and an unknown version.
    """

    where = f"F5 reproduction summary {source}"
    version = summary.get("schema_version")
    labelled = "kinetic_normalization" in summary
    if version == F5_REPRODUCTION_SCHEMA:
        if not labelled:
            raise ValueError(f"{where}: {version} has no kinetic_normalization label")
        label = summary["kinetic_normalization"]
        if label not in KINETIC_NORMALIZATION_LABELS:
            raise ValueError(f"{where}: kinetic_normalization {label!r} is not an allowed label")
        if label != KINETIC_NORMALIZATION:
            raise ValueError(
                f"{where}: F5 is a primary physical family and holds only "
                f"{KINETIC_NORMALIZATION!r}, found {label!r}"
            )
        return (label, False)
    if version == F5_LEGACY_REPRODUCTION_SCHEMA:
        if labelled:
            raise ValueError(f"{where}: legacy {version} carries a kinetic_normalization label")
        return ("legacy_c_half", True)
    if version in F5_A2_REPRODUCTION_SCHEMAS:
        if labelled:
            raise ValueError(f"{where}: legacy {version} carries a kinetic_normalization label")
        short = F5_A2_REPRODUCTION_SCHEMAS[version]
        if not match_f5_legacy_layouts(summary, short):
            unrecorded = _unrecorded_paths(summary, short)
            detail = (
                f"; paths in no recorded layout: {', '.join(unrecorded[:5])}"
                + (" …" if len(unrecorded) > 5 else "")
                if unrecorded
                else "; its path set matches no single recorded layout"
            )
            raise ValueError(f"{where}: {version} matches no recorded layout{detail}")
        return ("legacy_c_half", True)
    raise ValueError(f"{where}: unknown schema_version {version!r}")


# Derived merge JSON of merge_orientifold_shards.py.
SHARD_MERGE_SCHEMA = "cyaxiverse-orientifold-shard-merge-1.1"
LEGACY_SHARD_MERGE_SCHEMA = "cyaxiverse-orientifold-shard-merge-1.0"


def shard_merge_label(merged: dict[str, Any], source: Any) -> tuple[str, bool]:
    """Return ``(label, inferred)`` for a merge JSON of ``merge_orientifold_shards.py``.

    A merge is a derived output (CYAX-0232 R-006): it carries its inputs'
    common label.
    - ``shard-merge-1.1`` needs ``kinetic_normalization`` and
      ``kinetic_normalization_inferred``: ``"sugra_c1"`` with ``False``, or a
      derived ``"legacy_c_half"`` with ``True``.
    - ``shard-merge-1.0`` without a label: ``("legacy_c_half", True)``.

    Raises a ``ValueError`` naming ``source`` otherwise.
    """

    where = f"orientifold shard merge {source}"
    version = merged.get("schema_version")
    labelled = "kinetic_normalization" in merged or "kinetic_normalization_inferred" in merged
    if version == SHARD_MERGE_SCHEMA:
        label = merged.get("kinetic_normalization")
        inferred = merged.get("kinetic_normalization_inferred")
        if label not in KINETIC_NORMALIZATION_LABELS:
            raise ValueError(f"{where}: kinetic_normalization {label!r} is missing or not allowed")
        if not isinstance(inferred, bool):
            raise ValueError(f"{where}: kinetic_normalization_inferred is missing or not a bool")
        if (label, inferred) not in ((KINETIC_NORMALIZATION, False), ("legacy_c_half", True)):
            raise ValueError(
                f"{where}: a merge of physical F5 outputs holds {KINETIC_NORMALIZATION!r} "
                f"or an inferred 'legacy_c_half', found {label!r} (inferred={inferred})"
            )
        return (label, inferred)
    if version == LEGACY_SHARD_MERGE_SCHEMA:
        if labelled:
            raise ValueError(f"{where}: legacy {version} carries a kinetic_normalization label")
        return ("legacy_c_half", True)
    raise ValueError(f"{where}: unknown schema_version {version!r}")


def common_label(
    labels: Iterable[tuple[str, bool]], sources: Iterable[Any]
) -> tuple[str, bool]:
    """Return the one ``(label, inferred)`` pair shared by every input.

    Raises a ``ValueError`` naming both labels and their sources when the
    inputs differ (CYAX-0232 R-007 combination), or when there is no input.
    """

    first = None
    for label, source in zip(labels, sources):
        if first is None:
            first = (label, source)
        elif label != first[0]:
            raise ValueError(
                f"cannot combine kinetic_normalization {first[0][0]!r} ({first[1]}) "
                f"and {label[0]!r} ({source})"
            )
    if first is None:
        raise ValueError("no inputs to combine")
    return first[0]
