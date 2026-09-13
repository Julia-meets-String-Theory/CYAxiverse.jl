#!/usr/bin/env python3
"""Offline validation for the frozen CYAX-0155 source snapshot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SNAPSHOT_ID = "cyax-0155-adversarial-20260913T043555Z"


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def main() -> int:
    snapshot = json.loads((HERE / "source_snapshot.json").read_text())
    assert snapshot["snapshot_id"] == SNAPSHOT_ID
    assert snapshot["source_count"] == len(snapshot["sources"]) == 10
    ids = [source["id"] for source in snapshot["sources"]]
    assert len(ids) == len(set(ids))

    issue = json.loads((HERE / "sources/issue_155.json").read_text())
    timeline = json.loads((HERE / "sources/issue_155_timeline.json").read_text())
    pr = json.loads((HERE / "sources/pr_156.json").read_text())
    assert (issue["state"], issue["stateReason"]) == ("CLOSED", "COMPLETED")
    assert issue["projectItems"][0]["status"] == "Done"
    assert "Keep #155 open until" in issue["comments"][0]["remaining_project_action"]
    assert timeline == [{"event": "closed", "created_at": "2026-09-11T01:09:18Z", "actor": "vmmhep", "commit_id": None, "performed_via_github_app": None}]
    assert pr["state"] == "MERGED"
    assert pr["baseRefName"] == "vmm"
    assert pr["mergeCommit"] == "f0013552cd69a93221464e9c8ccfd56339f39052"

    audit = json.loads((HERE / "sources/source_search_audit.json").read_text())
    assert audit["github"]["issue_155"] == {
        "comments_reported": 1,
        "comments_pages": 1,
        "comments_captured": 1,
        "timeline_pages": 1,
        "timeline_events_captured": 10,
    }
    assert audit["github"]["pr_156"]["issue_comments_captured"] == 0
    assert audit["github"]["pr_156"]["reviews_captured"] == 0
    assert len(audit["timeline_events"]) == 10
    assert [e for e in audit["timeline_events"] if e["event"] == "closed"] == [{
        "id": 30941590523,
        "event": "closed",
        "created_at": "2026-09-11T01:09:18Z",
        "actor": "vmmhep",
        "commit_id": None,
        "performed_via_github_app": None,
    }]
    refs = (HERE / "sources/public_ref_inventory.txt").read_bytes()
    assert refs.count(b"\n") == audit["git"]["ref_count"] == 66
    assert hashlib.sha256(refs).hexdigest() == audit["git"]["ref_inventory_sha256"]

    commit = pr["mergeCommit"]
    expected = {
        "specs/0155-private-safe-chat-checkpoints/spec.md": "4e010cd612eae44cb75eb729078c92a73476d2ec6989b04c5112854977d43363",
        "AGENTS.md": "b213b788da72e703363810be998352fdf5e91a28991a8bcbeb7860213f642e32",
        ".agents/skills/cyaxiverse-sdd/SKILL.md": "235a0d4d16a0545af6af34e2b26b7f996e186e2816c6aef134a5bf7cb897784b",
        "docs/PRIVACY_AND_CONVERSATION_CHECKPOINTS.md": "c7f4bc8242b29928ded4762317c52ee7a0a10eb9f7260d0098c886297397214f",
    }
    for path, digest in expected.items():
        assert hashlib.sha256(git_bytes(commit, path)).hexdigest() == digest

    assert subprocess.check_output(["git", "cat-file", "-t", commit], cwd=ROOT, text=True).strip() == "commit"
    print(f"PASS snapshot {SNAPSHOT_ID}: 10 sources; GitHub fixtures and four revision-pinned repository artifacts verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
