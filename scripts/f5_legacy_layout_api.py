"""A2 normalization and the layout type for the F5 legacy layout records (CYAX-0232 T041, T061).

The recorded layouts are in `f5_legacy_layouts_v10a.py`, `_v10b.py` and `_v11.py`;
their prose records are in `specs/0232-kinetic-normalization/layouts/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Layout:
    id: str                  # e.g. "1.0-G2-L05" (unique across all records)
    version: str             # "1.0" or "1.1"
    group: str               # e.g. "1.0-G2"
    commits: tuple[str, ...] # every commit of the group (short ids)
    flags: tuple[str, ...]   # the CLI flags on for this layout, sorted
    h11_family: str          # e.g. "T" (h11 has a Table 1 row), "N" (no row), "P" (preflight h11)
    required: frozenset[str]
    optional: frozenset[str]
    map_nodes: frozenset[str]


def normalize(obj: Any, map_nodes: Iterable[str]) -> frozenset[str]:
    """Return the A2 normalized path set of a JSON value.

    - A dict key `k` under prefix `p` gives the path `p.k` (or `k` at the top),
      whatever its value, including null.
    - A dict at a path listed in `map_nodes` is a map node: its keys give no
      path; each of its values is normalized under the prefix `p.<map>`. No
      bare `p.<map>` path is emitted.
    - A list at prefix `p` gives no path of its own (its key already did). Each
      dict element is normalized under `p[]`, as a union over elements; a list
      element is itself an array, normalized under `p[]` the same way (so a
      list of lists of dicts gives `p[][].k`). Scalar and null elements add
      nothing.
    - Scalars and null add nothing below their key.
    """

    nodes = frozenset(map_nodes)
    paths: set[str] = set()

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            if prefix in nodes:
                for item in value.values():
                    walk(item, prefix + ".<map>")
                return
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                paths.add(path)
                walk(item, path)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    walk(item, prefix + "[]")

    walk(obj, "")
    return frozenset(paths)


def synthesize(paths: Iterable[str]) -> dict[str, Any]:
    """Build a JSON value whose normalized path set is exactly `paths`.

    Segments: `name` (dict key), `name[]` handled as the key `name` holding a
    list with one dict element (`name[][]`: a list holding one such list), and
    `<map>` as a dict with one data key
    (`"k0"`). A path with nothing below it gets the value null. Used to
    self-check an encoding: normalize(synthesize(S), layout.map_nodes) == S.
    """

    root: dict[str, Any] = {}
    for path in sorted(paths, key=lambda p: p.count(".")):
        node: Any = root
        segments = path.split(".")
        for index, segment in enumerate(segments):
            last = index == len(segments) - 1
            if segment == "<map>":
                if not isinstance(node, dict):
                    raise ValueError(f"bad map segment in {path}")
                node = node.setdefault("k0", {})
                continue
            depth = 0
            key = segment
            while key.endswith("[]"):
                key, depth = key[:-2], depth + 1
            if depth:
                current = node.get(key)
                if not isinstance(current, list):
                    current = [{}]
                    for _ in range(depth - 1):
                        current = [current]
                    node[key] = current
                for _ in range(depth):
                    current = current[0]
                node = current
            else:
                if last:
                    node.setdefault(key, None)
                else:
                    current = node.get(key)
                    if not isinstance(current, dict):
                        current = {}
                        node[key] = current
                    node = current
    return root
