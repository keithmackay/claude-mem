#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["chromadb>=1.0.0"]
# ///
"""
Backfill embeddings for all existing claude-mem SQLite data into ChromaDB,
AND backfill verbatim assistant responses from Claude Code session JSONL files.

Uses Ollama's nomic-embed-text model (free, local, 768-dim).

Usage:
  uvx --python 3.13 --with chromadb python scripts/backfill_embeddings.py

Options (env vars):
  CHROMA_DATA_DIR     path to ChromaDB storage (default: ~/.claude-mem/vector-db)
  OLLAMA_BASE_URL     Ollama server (default: http://localhost:11434)
  OLLAMA_EMBED_MODEL  model name (default: nomic-embed-text)
  CLAUDE_MEM_DB       path to SQLite DB (default: ~/.claude-mem/claude-mem.db)
  CLAUDE_PROJECTS_DIR path to Claude Code session files (default: ~/.claude/projects)
  PROJECT             restrict to one project (default: all)
  BATCH_SIZE          docs per Chroma batch (default: 100)
  DRY_RUN             set to '1' to print counts without syncing
  SKIP_RESPONSES      set to '1' to skip assistant response backfill
  SKIP_EMBEDDINGS     set to '1' to skip Chroma embedding backfill
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOME = Path.home()
DATA_DIR        = os.environ.get("CHROMA_DATA_DIR",     str(HOME / ".claude-mem" / "vector-db"))
DB_PATH         = os.environ.get("CLAUDE_MEM_DB",        str(HOME / ".claude-mem" / "claude-mem.db"))
PROJECTS_DIR    = os.environ.get("CLAUDE_PROJECTS_DIR",  str(HOME / ".claude" / "projects"))
OLLAMA_BASE     = os.environ.get("OLLAMA_BASE_URL",      "http://localhost:11434")
OLLAMA_URL      = OLLAMA_BASE + "/api/embed"   # batch endpoint, supports all models
MODEL           = os.environ.get("OLLAMA_EMBED_MODEL",   "all-minilm")
FILTER_PROJECT  = os.environ.get("PROJECT", "")
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", "100"))
DRY_RUN         = os.environ.get("DRY_RUN", "0") == "1"
SKIP_RESPONSES  = os.environ.get("SKIP_RESPONSES", "0") == "1"
SKIP_EMBEDDINGS = os.environ.get("SKIP_EMBEDDINGS", "0") == "1"

# Map project name → Claude Code project directory name
# Add entries here if the directory name doesn't match the project name.
PROJECT_DIR_MAP: dict[str, str] = {
    "openclaw": "-Users-keithmackay1--openclaw-workspace",
    "nanobot":  "-Users-keithmackay1-Projects-nanobot",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_CHARS = 500  # all-minilm has 256-token limit; dense text (paths) hits it ~600 chars

def embed_batch(texts: list[str], timeout: int = 300, retries: int = 3) -> list[list[float]]:
    """Embed texts via Ollama /api/embed (supports all models, accepts array input)."""
    truncated = [t[:MAX_CHARS] for t in texts]
    payload = json.dumps({"model": MODEL, "input": truncated}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())["embeddings"]
        except TimeoutError:
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"\n  [embed] timeout, retrying in {wait}s (attempt {attempt+1}/{retries})...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("embed_batch: unreachable")


def warmup_ollama() -> None:
    """Ensure Ollama has the model loaded before batch processing."""
    print(f"Warming up {MODEL} in Ollama...", flush=True)
    embed_batch(["warmup"])
    print("Ollama ready.", flush=True)


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

def extract_assistant_text(content: object) -> str:
    """Extract plain text from a claude message content field."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            c.get("text", "")
            for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        return " ".join(p for p in parts if p).strip()
    return ""


def iter_jsonl_responses(jsonl_path: Path) -> Iterator[dict]:
    """Yield assistant turns from a Claude Code session JSONL file.

    Each yielded dict has: session_id, message_uuid, parent_uuid,
    timestamp, response_text.
    """
    with open(jsonl_path, encoding="utf-8", errors="replace") as f:
        prompt_counter = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = d.get("message") or {}
            if d.get("type") == "user" and msg.get("role") == "user":
                prompt_counter += 1
                continue

            if d.get("type") != "assistant" or msg.get("role") != "assistant":
                continue

            text = extract_assistant_text(msg.get("content", []))
            if not text:
                continue

            # Skip pure tool-call turns (no real text to index)
            if text.startswith("<tool_call>"):
                continue

            yield {
                "session_id": d.get("sessionId", ""),
                "message_uuid": d.get("uuid", ""),
                "parent_uuid": d.get("parentUuid", ""),
                "timestamp": d.get("timestamp", ""),
                "prompt_number": prompt_counter,
                "response_text": text,
            }


def ensure_assistant_responses_table(db: sqlite3.Connection) -> None:
    """Create assistant_responses table if it doesn't exist (idempotent)."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS assistant_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claude_session_id TEXT NOT NULL,
            prompt_number INTEGER NOT NULL,
            response_text TEXT NOT NULL,
            message_uuid TEXT,
            created_at TEXT NOT NULL,
            created_at_epoch INTEGER NOT NULL,
            FOREIGN KEY(claude_session_id) REFERENCES sdk_sessions(claude_session_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_assistant_responses_session
            ON assistant_responses(claude_session_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS assistant_responses_fts USING fts5(
            response_text,
            content='assistant_responses',
            content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS assistant_responses_fts_insert
            AFTER INSERT ON assistant_responses BEGIN
                INSERT INTO assistant_responses_fts(rowid, response_text)
                VALUES (new.id, new.response_text);
            END;
        CREATE TRIGGER IF NOT EXISTS assistant_responses_fts_delete
            AFTER DELETE ON assistant_responses BEGIN
                INSERT INTO assistant_responses_fts(assistant_responses_fts, rowid, response_text)
                VALUES ('delete', old.id, old.response_text);
            END;
    """)
    db.commit()


def backfill_assistant_responses(db: sqlite3.Connection, project: str) -> int:
    """
    Parse Claude Code JSONL files for this project and store assistant responses
    in the assistant_responses table (verbatim, with FTS5 indexing).
    Returns the number of new rows inserted.
    """
    ensure_assistant_responses_table(db)

    dir_name = PROJECT_DIR_MAP.get(project)
    if not dir_name:
        print(f"  [assistant_responses] No JSONL dir mapped for project '{project}' — skipping")
        return 0

    jsonl_dir = Path(PROJECTS_DIR) / dir_name
    if not jsonl_dir.exists():
        print(f"  [assistant_responses] Dir not found: {jsonl_dir} — skipping")
        return 0

    # Load known session IDs for this project
    rows = db.execute(
        "SELECT claude_session_id FROM sdk_sessions WHERE project = ?", (project,)
    ).fetchall()
    known_sessions = {r[0] for r in rows}

    # Load already-stored message UUIDs to avoid duplicates
    existing_uuids: set[str] = set()
    rows = db.execute(
        """
        SELECT ar.message_uuid
        FROM assistant_responses ar
        JOIN sdk_sessions s ON ar.claude_session_id = s.claude_session_id
        WHERE s.project = ? AND ar.message_uuid IS NOT NULL AND ar.message_uuid != ''
        """,
        (project,),
    ).fetchall()
    for r in rows:
        existing_uuids.add(r[0])

    inserted = 0
    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    print(f"  [assistant_responses] scanning {len(jsonl_files)} JSONL files...", flush=True)

    for i, path in enumerate(jsonl_files):
        session_id = path.stem  # filename without .jsonl
        if session_id not in known_sessions:
            continue

        if i % 500 == 0 and i > 0:
            print(f"    progress: {i}/{len(jsonl_files)} files, {inserted} rows inserted", flush=True)

        for turn in iter_jsonl_responses(path):
            if turn["message_uuid"] and turn["message_uuid"] in existing_uuids:
                continue

            if DRY_RUN:
                inserted += 1
                continue

            ts = turn["timestamp"] or ""
            try:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                epoch = int(dt.timestamp())
            except Exception:
                epoch = 0

            db.execute(
                """
                INSERT INTO assistant_responses
                    (claude_session_id, prompt_number, response_text, message_uuid, created_at, created_at_epoch)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, turn["prompt_number"], turn["response_text"],
                 turn["message_uuid"], ts, epoch),
            )
            if turn["message_uuid"]:
                existing_uuids.add(turn["message_uuid"])
            inserted += 1

    if not DRY_RUN:
        db.commit()

    return inserted


def batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def get_existing_ids(col, doc_type: str) -> set[int]:
    """Fetch all sqlite_id values from a Chroma collection for a given doc_type."""
    existing: set[int] = set()
    offset = 0
    while True:
        result = col.get(
            where={"doc_type": doc_type},
            limit=1000,
            offset=offset,
            include=["metadatas"],
        )
        metas = result.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            if m and m.get("sqlite_id"):
                existing.add(int(m["sqlite_id"]))
        offset += len(metas)
        if len(metas) < 1000:
            break
    return existing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import chromadb

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    # Discover projects
    if FILTER_PROJECT:
        projects = [FILTER_PROJECT]
    else:
        rows = db.execute("SELECT DISTINCT project FROM sdk_sessions ORDER BY project").fetchall()
        projects = [r["project"] for r in rows]

    print(f"Projects to backfill: {projects}")
    print(f"Embedding model: {MODEL} via {OLLAMA_URL}")
    print(f"ChromaDB path: {DATA_DIR}")
    if DRY_RUN:
        print("[DRY RUN — no writes]")
    print()

    warmup_ollama()
    client = chromadb.PersistentClient(path=DATA_DIR)
    total_added = 0
    t0 = time.monotonic()

    for project in projects:
        col_name = f"cm__{project}"
        print(f"=== Project: {project} (collection: {col_name}) ===")

        if not DRY_RUN:
            col = client.get_or_create_collection(
                col_name,
                metadata={"embedding_model": MODEL, "embedding_provider": "ollama"},
            )

        # ---- User prompts ------------------------------------------------
        rows = db.execute("""
            SELECT up.id, up.prompt_text, up.prompt_number, up.created_at_epoch,
                   s.sdk_session_id
            FROM user_prompts up
            JOIN sdk_sessions s ON up.claude_session_id = s.claude_session_id
            WHERE s.project = ?
            ORDER BY up.id
        """, (project,)).fetchall()
        print(f"  user_prompts: {len(rows)} total")

        if not DRY_RUN and rows:
            existing = get_existing_ids(col, "user_prompt")
            missing = [r for r in rows if r["id"] not in existing]
            print(f"  user_prompts: {len(existing)} already indexed, {len(missing)} to add")

            added = 0
            for batch in batched(missing, BATCH_SIZE):
                texts  = [r["prompt_text"] for r in batch]
                ids    = [f"prompt_{r['id']}" for r in batch]
                metas  = [{
                    "sqlite_id": r["id"],
                    "doc_type":  "user_prompt",
                    "sdk_session_id": r["sdk_session_id"],
                    "project": project,
                    "created_at_epoch": r["created_at_epoch"],
                    "prompt_number": r["prompt_number"],
                } for r in batch]
                embs = embed_batch(texts)
                col.add(documents=texts, embeddings=embs, ids=ids, metadatas=metas)
                added += len(batch)
                total_added += len(batch)
                elapsed = time.monotonic() - t0
                rate = total_added / elapsed if elapsed > 0 else 0
                print(f"    prompts: {added}/{len(missing)}  ({rate:.1f} docs/s)", end="\r")
            print(f"    prompts: {added} added{'':20}")

        # ---- Observations ------------------------------------------------
        rows = db.execute("""
            SELECT id, sdk_session_id, type, title, subtitle, narrative, text,
                   facts, concepts, files_read, files_modified, created_at_epoch, prompt_number
            FROM observations WHERE project = ? ORDER BY id
        """, (project,)).fetchall()
        print(f"  observations: {len(rows)} total")

        if not DRY_RUN and rows:
            existing = get_existing_ids(col, "observation")
            missing = [r for r in rows if r["id"] not in existing]
            print(f"  observations: {len(existing)} already indexed, {len(missing)} to add")

            added = 0
            for obs in missing:
                docs, ids, metas = [], [], []
                base = {
                    "sqlite_id": obs["id"],
                    "doc_type": "observation",
                    "sdk_session_id": obs["sdk_session_id"],
                    "project": project,
                    "created_at_epoch": obs["created_at_epoch"],
                    "type": obs["type"] or "change",
                    "title": obs["title"] or "",
                }
                if obs["narrative"]:
                    docs.append(obs["narrative"]); ids.append(f"obs_{obs['id']}_narrative"); metas.append({**base, "field_type": "narrative"})
                if obs["text"]:
                    docs.append(obs["text"]); ids.append(f"obs_{obs['id']}_text"); metas.append({**base, "field_type": "text"})
                for i, fact in enumerate(json.loads(obs["facts"] or "[]")):
                    docs.append(fact); ids.append(f"obs_{obs['id']}_fact_{i}"); metas.append({**base, "field_type": "fact", "fact_index": i})
                if docs:
                    embs = embed_batch(docs)
                    col.add(documents=docs, embeddings=embs, ids=ids, metadatas=metas)
                added += 1
                total_added += len(docs)
                if added % 10 == 0:
                    print(f"    observations: {added}/{len(missing)}", end="\r")
            print(f"    observations: {added} added{'':20}")

        # ---- Session summaries -------------------------------------------
        rows = db.execute("""
            SELECT id, sdk_session_id, request, investigated, learned, completed,
                   next_steps, notes, created_at_epoch, prompt_number
            FROM session_summaries WHERE project = ? ORDER BY id
        """, (project,)).fetchall()
        print(f"  summaries: {len(rows)} total")

        if not DRY_RUN and rows:
            existing = get_existing_ids(col, "session_summary")
            missing = [r for r in rows if r["id"] not in existing]
            print(f"  summaries: {len(existing)} already indexed, {len(missing)} to add")

            added = 0
            for s in missing:
                docs, ids, metas = [], [], []
                base = {
                    "sqlite_id": s["id"],
                    "doc_type": "session_summary",
                    "sdk_session_id": s["sdk_session_id"],
                    "project": project,
                    "created_at_epoch": s["created_at_epoch"],
                    "prompt_number": s["prompt_number"] or 0,
                }
                for field in ("request", "investigated", "learned", "completed", "next_steps", "notes"):
                    val = s[field]
                    if val:
                        docs.append(val); ids.append(f"summary_{s['id']}_{field}"); metas.append({**base, "field_type": field})
                if docs:
                    embs = embed_batch(docs)
                    col.add(documents=docs, embeddings=embs, ids=ids, metadatas=metas)
                added += 1
                total_added += len(docs)
            print(f"    summaries: {added} added")

        # ---- Assistant responses (SQLite verbatim + Chroma embeddings) ---
        if not SKIP_RESPONSES:
            n_inserted = backfill_assistant_responses(db, project)
            print(f"  assistant_responses: {n_inserted} new rows inserted")

        if not DRY_RUN and not SKIP_EMBEDDINGS:
            resp_rows = db.execute("""
                SELECT ar.id, ar.claude_session_id, ar.prompt_number,
                       ar.response_text, ar.created_at_epoch,
                       s.sdk_session_id
                FROM assistant_responses ar
                JOIN sdk_sessions s ON ar.claude_session_id = s.claude_session_id
                WHERE s.project = ?
                ORDER BY ar.id
            """, (project,)).fetchall()
            print(f"  assistant_responses: {len(resp_rows)} total in SQLite")

            if resp_rows:
                existing = get_existing_ids(col, "assistant_response")
                missing_resp = [r for r in resp_rows if r["id"] not in existing]
                print(f"  assistant_responses: {len(existing)} already indexed, {len(missing_resp)} to add")

                added = 0
                for batch in batched(missing_resp, BATCH_SIZE):
                    texts = [r["response_text"] for r in batch]
                    ids   = [f"response_{r['id']}" for r in batch]
                    metas = [{
                        "sqlite_id": r["id"],
                        "doc_type": "assistant_response",
                        "sdk_session_id": r["sdk_session_id"],
                        "project": project,
                        "created_at_epoch": r["created_at_epoch"] or 0,
                        "prompt_number": r["prompt_number"] or 0,
                    } for r in batch]
                    embs = embed_batch(texts)
                    col.add(documents=texts, embeddings=embs, ids=ids, metadatas=metas)
                    added += len(batch)
                    total_added += len(batch)
                    elapsed = time.monotonic() - t0
                    rate = total_added / elapsed if elapsed > 0 else 0
                    print(f"    responses: {added}/{len(missing_resp)}  ({rate:.1f} docs/s)", end="\r")
                print(f"    responses: {added} added{'':20}")

        print()

    elapsed = time.monotonic() - t0
    rate = total_added / elapsed if elapsed > 0 else 0
    print(f"Backfill complete: {total_added} documents in {elapsed:.1f}s ({rate:.1f} docs/s)")
    db.close()


if __name__ == "__main__":
    main()
