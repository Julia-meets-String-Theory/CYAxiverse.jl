# Privacy-safe conversation checkpoints

This guide explains how exploratory AI/chat work may become visible in the CYAxiverse GitHub Project without turning GitHub into an archive or index of private conversations.

`AGENTS.md` is the normative repository contract. `cyaxiverse-sdd` contains the operational SDD procedure. This document is human-facing guidance.

## The boundary

Treat private chats, local agent sessions, local filesystem context, connected-app context, and private attachments as **non-public by default**.

Before anything derived from that context is written durably to GitHub, sanitize it. Do not publish, unless the owner explicitly approves the exact datum:

- private conversation/share URLs or transcript dumps;
- absolute local filesystem paths or home-directory structure;
- local usernames, hostnames, machine names, or device identifiers;
- local Codex/agent/workspace/session paths;
- private attachment locations or connector-local references;
- secrets, credentials, or tokens.

Use repository-relative paths for repository files. If there is doubt about whether something is safe to publish, omit it and ask before publishing.

This forward-looking rule does not by itself remove information already present in Git history, Issues, PRs, Actions logs, or artifacts. Historical leakage should be handled by a separate read-only audit and an explicit remediation decision.

## What a conversation checkpoint is

A conversation checkpoint is a **sanitized project summary**, not a transcript and not a link back to the private conversation.

Promote a discussion to a durable Project item only when it has produced something worth tracking independently, for example:

- a research question that may continue across sessions;
- an owner decision or unresolved decision;
- a blocker or reproducible mismatch;
- a proposed S1–S3 specification;
- an implementation follow-up;
- an evidence-bearing investigation whose state should survive the chat.

Casual discussion, brainstorming with no durable outcome, and intermediate reasoning should stay out of GitHub.

## Safe checkpoint contents

A checkpoint may contain:

```text
Title / research question
Workstream
Status
Priority, when useful
Public-safe current conclusion or decision
Next action / owner decision needed
Related Issue, spec, PR, or repository-relative file
Optional opaque private reference
```

An opaque private reference must not resolve from GitHub to a private chat, local path, attachment, account, or machine. Any resolving mapping stays outside GitHub.

## Kanban model

Conversation-derived work uses the same Project lifecycle as the rest of CYAxiverse:

`Backlog → Specifying → Spec Review → Ready → Implementing → Verification → Done`

Do not create a second conversation-specific lifecycle. For S1–S3 work, the governing Issue remains the primary Project card; linked PRs provide implementation/integration/evidence rather than duplicate WIP cards.

The intended **Research & Chats** saved view is therefore another view over the same Project items, grouped by the normal `Status` field. It should expose only sanitized project metadata. A lightweight non-private marker/filter for conversation-checkpoint Issues may be added in Project configuration, but it must not encode a chat URL or other private locator.

This means the board can answer “what investigations are active, what did we conclude, and what happens next?” without answering “where is the private transcript?”

## Suggested working convention

At the point an exploratory conversation becomes durable, create or update the governing Issue rather than creating a card for every chat. Record the smallest public-safe checkpoint that lets future work resume from durable state.

A useful shorthand request is **“checkpoint this to the board”**. The resulting GitHub write should still pass the privacy boundary above; the phrase is never permission to copy the conversation verbatim.

If the conversation advances an existing governing Issue, update that Issue instead of creating a duplicate card. Create a new Issue only when the outcome is independently durable or separately trackable.
