#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["chromadb>=1.0.0"]
# ///
"""
Backfill embeddings for all existing claude-mem SQLite data into ChromaDB.

Uses Ollama's nomic-embed-text model (free, local, 768-dim).

Usage:
  uvx --python 3.13 --with chromadb python scripts/backfill_embeddings.py

Options (env vars):
  CHROMA_DATA_DIR   path to ChromaDB storage (default: ~/.claude-mem/vector-db)
  OLLAMA_BASE_URL   Ollama server (default: http://localhost:11434)
  OLLAMA_EMBED_MODEL  model name (default: nomic-embed-text)
  CLAUDE_MEM_DB     path to SQLite DB (default: ~/.claude-mem/claude-mem.db)
  PROJECT           restrict to one project (default: all)
  BATCH_SIZE        docs per Chroma batch (default: 100)
  DRY_RUN           set to '1' to print counts without syncing
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
DATA_DIR   = os.environ.get("CHROMA_DATA_DIR",    str(HOME / ".claude-mem" / "vector-db"))
DB_PATH    = os.environ.get("CLAUDE_MEM_DB",       str(HOME / ".claude-mem" / "claude-mem.db"))
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL",     "http://localhost:11434") + "/api/embeddings"
MODEL      = os.environ.get("OLLAMA_EMBED_MODEL",  "nomic-embed-text")
FILTER_PROJECT = os.environ.get("PROJECT", "")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
DRY_RUN    = os.environ.get("DRY_RUN", "0") == "1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str], timeout: int = 120) -> list[list[float]]:
    """Embed a list of texts via Ollama HTTP API."""
    result = []
    for text in texts:
        payload = json.dumps({"model": MODEL, "prompt": text}).encode()
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result.append(json.loads(resp.read())["embedding"])
    return result


def warmup_ollama() -> None:
    """Ensure Ollama has the model loaded before batch processing."""
    print(f"Warming up {MODEL} in Ollama...", flush=True)
    embed_batch(["warmup"], timeout=120)
    print("Ollama ready.", flush=True)


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

        print()

    elapsed = time.monotonic() - t0
    rate = total_added / elapsed if elapsed > 0 else 0
    print(f"Backfill complete: {total_added} documents in {elapsed:.1f}s ({rate:.1f} docs/s)")
    db.close()


if __name__ == "__main__":
    main()
