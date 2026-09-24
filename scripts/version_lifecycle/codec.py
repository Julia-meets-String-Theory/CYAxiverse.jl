"""Canonical JSON and digest primitives for immutable lifecycle artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Return the repository's canonical JSON encoding for *value*.

    ``json.dumps`` is used with an explicit configuration so this function is
    stable across callers.  Lifecycle artifacts use the printable ASCII JSON
    subset: nulls, floats, controls and non-ASCII strings are rejected.  The
    returned artifact has no trailing LF.
    """

    _validate_json_value(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not canonical JSON data: {error}") from error
    return encoded.encode("utf-8")


def _validate_json_value(value: Any) -> None:
    """Reject Python values whose implicit JSON coercion is ambiguous."""

    if value is None:
        raise ValueError("null is not permitted in canonical lifecycle JSON")
    if isinstance(value, str):
        if any(ord(char) < 0x20 or ord(char) > 0x7E for char in value):
            raise ValueError("canonical lifecycle JSON strings must be printable ASCII")
        return
    if isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        raise ValueError("floating-point values are not permitted in canonical lifecycle JSON")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            if any(ord(char) < 0x20 or ord(char) > 0x7E for char in key):
                raise ValueError("canonical lifecycle JSON object keys must be printable ASCII")
            _validate_json_value(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _validate_json_value(child)
        return
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def sha256_hex(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact *value* bytes."""

    if not isinstance(value, bytes):
        raise TypeError("sha256_hex requires exact bytes")
    return hashlib.sha256(value).hexdigest()


__all__ = ["canonical_json", "sha256_hex"]
