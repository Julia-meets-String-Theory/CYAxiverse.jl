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
# Inferred only for a file that matches a recorded layout (Clarification A2);
# the layout records and that matching are not implemented yet.
F5_A2_REPRODUCTION_SCHEMAS = (
    "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.0",
    "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.1",
)


def f5_reproduction_label(summary: dict[str, Any], source: Any) -> tuple[str, bool]:
    """Return ``(label, inferred)`` for an F5 reproduction JSON summary.

    - ``…-reproduction-1.3`` needs the top-level label ``"sugra_c1"`` (F5 is a
      primary physical family): ``("sugra_c1", False)``.
    - ``…-reproduction-1.2`` without a label: ``("legacy_c_half", True)``.

    Raises a ``ValueError`` naming ``source`` for a missing or wrong label, a
    legacy version that carries a label, an unknown version, and for
    ``…-1.0`` and ``…-1.1``, which are not inferred until the A2 layout
    matching exists.
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
        raise ValueError(
            f"{where}: {version} is inferred only by the A2 layout records, "
            "which are not implemented yet"
        )
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
