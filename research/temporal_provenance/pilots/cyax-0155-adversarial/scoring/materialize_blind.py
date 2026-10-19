#!/usr/bin/env python3
"""Create scorer-facing opaque response copies with citation labels neutralized."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
OUT = HERE / "blind_responses"
SOURCE_IDS = [
    "issue-155-current", "issue-155-checkpoint", "issue-155-closure-event",
    "pr-156", "pr-156-merge-commit", "spec-0155", "agents-contract-at-merge",
    "sdd-contract-at-merge", "checkpoint-guide-at-merge", "closure-search",
]


def neutralize(text: str) -> str:
    text = re.sub(r"\[(?:A\d{2})(?:[–—,-](?:A)?\d{2})?(?:,?\s*A\d{2})*\]", "[source-ref]", text)
    for source in sorted(SOURCE_IDS, key=len, reverse=True):
        text = text.replace(source, "source-ref")
    text = re.sub(r"(?:source-ref[;,]?\s*){2,}", "source-ref ", text)
    return text


def main() -> int:
    mapping = json.loads((HERE / "condition_mapping.json").read_text())["mapping"]
    OUT.mkdir(exist_ok=True)
    manifest = {"response_freeze": "fd8d9313d61445e430fb796cc7a3e2935b7886a9", "items": {}}
    for opaque, run in sorted(mapping.items()):
        raw = (PILOT / "runs" / run.lower() / "response.md").read_text()
        blinded = neutralize(raw)
        if re.search(r"\b(?:A[1-3]|B[1-3]|Condition A|Condition B)\b", blinded, re.I):
            raise ValueError(f"condition label survived in {opaque}")
        path = OUT / f"{opaque}.md"
        path.write_text(blinded)
        manifest["items"][opaque] = {
            "words": len(blinded.split()),
            "bytes": len(blinded.encode()),
            "sha256": hashlib.sha256(blinded.encode()).hexdigest(),
        }
    (HERE / "blind_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
