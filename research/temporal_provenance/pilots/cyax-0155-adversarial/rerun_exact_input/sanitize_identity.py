#!/usr/bin/env python3
"""Deterministically remove local Codex identity values from event evidence.

Identity fields remain in the event structure so event semantics and the
no-tool audit are preserved.  Their values become labelled SHA-256
pseudonyms.  The original event stream is not retained; callers record its
byte count and SHA-256 as provenance before writing the sanitized stream.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SCHEMA = "cyax-0163-identity-sanitization-v1"
DOMAIN = b"cyax-0163-local-codex-identity-v1:"
IDENTITY_KEYS = ("thread_id", "session_id", "conversation_id", "turn_id")
PSEUDONYM_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pseudonym(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("identity value must be a non-empty string")
    return "sha256:" + hashlib.sha256(DOMAIN + value.encode("utf-8")).hexdigest()


def is_pseudonym(value: object) -> bool:
    return isinstance(value, str) and bool(PSEUDONYM_RE.fullmatch(value))


def _sanitize(value: Any, identities: set[str]) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if key in IDENTITY_KEYS and isinstance(child, str) and child:
                result[key] = child if is_pseudonym(child) else pseudonym(child)
                identities.add(result[key])
            else:
                result[key] = _sanitize(child, identities)
        return result
    if isinstance(value, list):
        return [_sanitize(child, identities) for child in value]
    return value


def sanitize_event_stream(data: bytes) -> tuple[bytes, dict[str, Any]]:
    """Return sanitized JSONL bytes and non-sensitive provenance metadata."""
    output: list[bytes] = []
    identities: set[str] = set()
    lines = data.splitlines(keepends=True)
    if not lines:
        raise ValueError("event stream is empty")
    for index, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip(b"\r\n")
        ending = raw_line[len(line):]
        if not line:
            output.append(raw_line)
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"event line {index} is not JSON") from exc
        sanitized = _sanitize(event, identities)
        encoded = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        output.append(encoded + ending)
    sanitized_bytes = b"".join(output)
    metadata = {
        "schema": SCHEMA,
        "identity_fields": list(IDENTITY_KEYS),
        "identity_value_encoding": "sha256:<64 lowercase hex> with a fixed domain prefix",
        "original_bytes": len(data),
        "original_sha256": sha256(data),
        "sanitized_bytes": len(sanitized_bytes),
        "sanitized_sha256": sha256(sanitized_bytes),
        "hashed_identity_count": len(identities),
        "hashed_identities": sorted(identities),
    }
    verify_sanitized_event_stream(sanitized_bytes)
    return sanitized_bytes, metadata


def verify_sanitized_event_stream(data: bytes) -> None:
    """Reject any non-pseudonym value in an identity field."""
    if not data:
        raise ValueError("sanitized event stream is empty")
    for index, line in enumerate(data.splitlines(), 1):
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"sanitized event line {index} is not JSON") from exc
        _verify_value(event, index)


def _verify_value(value: Any, line: int) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in IDENTITY_KEYS and child is not None and not is_pseudonym(child):
                raise ValueError(f"line {line}: identity field {key} is not a SHA-256 pseudonym")
            _verify_value(child, line)
    elif isinstance(value, list):
        for child in value:
            _verify_value(child, line)
