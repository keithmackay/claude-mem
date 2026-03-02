#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["chromadb>=1.0.0", "mcp>=1.0.0"]
# ///
"""
Chroma MCP server with Ollama embeddings.

Drop-in replacement for chroma-mcp that uses nomic-embed-text (or any other
Ollama model) for embeddings instead of the default sentence-transformers model.
Embeddings are computed via the Ollama HTTP API and stored as explicit vectors
in ChromaDB, so no Python embedding dependencies (onnxruntime, torch, etc.) are needed.

Implements the same MCP tool interface as chroma-mcp:
  - chroma_get_collection_info
  - chroma_create_collection
  - chroma_add_documents
  - chroma_query_documents
  - chroma_get_documents
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any

import chromadb
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ---------------------------------------------------------------------------
# Configuration (from environment / defaults)
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get("CHROMA_DATA_DIR", os.path.expanduser("~/.claude-mem/vector-db"))
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_EMBED_URL = f"{OLLAMA_BASE}/api/embeddings"

# ---------------------------------------------------------------------------
# ChromaDB client (persistent, path = DATA_DIR)
# ---------------------------------------------------------------------------

_client: chromadb.PersistentClient | None = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=DATA_DIR)
    return _client


# ---------------------------------------------------------------------------
# Ollama embedding helper
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Compute embeddings for a list of texts using Ollama."""
    embeddings: list[list[float]] = []
    for text in texts:
        payload = json.dumps({"model": OLLAMA_MODEL, "prompt": text}).encode()
        req = urllib.request.Request(
            OLLAMA_EMBED_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        embeddings.append(data["embedding"])
    return embeddings


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

app = Server("chroma-ollama-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="chroma_get_collection_info",
            description="Get info about a Chroma collection",
            inputSchema={
                "type": "object",
                "properties": {"collection_name": {"type": "string"}},
                "required": ["collection_name"],
            },
        ),
        Tool(
            name="chroma_create_collection",
            description="Create a Chroma collection (embeddings via Ollama)",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string"},
                    "embedding_function_name": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["collection_name"],
            },
        ),
        Tool(
            name="chroma_add_documents",
            description="Add documents to a Chroma collection",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string"},
                    "documents": {"type": "array", "items": {"type": "string"}},
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "metadatas": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["collection_name", "documents", "ids"],
            },
        ),
        Tool(
            name="chroma_query_documents",
            description="Query a Chroma collection for similar documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string"},
                    "query_texts": {"type": "array", "items": {"type": "string"}},
                    "n_results": {"type": "integer"},
                    "where": {"type": "object"},
                    "where_document": {"type": "object"},
                    "include": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["collection_name", "query_texts"],
            },
        ),
        Tool(
            name="chroma_get_documents",
            description="Get documents from a Chroma collection",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string"},
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                    "where": {"type": "object"},
                    "include": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["collection_name"],
            },
        ),
    ]


def _text(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(data, default=str))]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    client = get_client()

    # ------------------------------------------------------------------
    if name == "chroma_get_collection_info":
        col = client.get_collection(arguments["collection_name"])
        return _text({"name": col.name, "count": col.count(), "metadata": col.metadata})

    # ------------------------------------------------------------------
    if name == "chroma_create_collection":
        col_name = arguments["collection_name"]
        metadata = arguments.get("metadata") or {}
        # Store embedding model in collection metadata so it self-documents
        metadata["embedding_model"] = OLLAMA_MODEL
        col = client.get_or_create_collection(col_name, metadata=metadata)
        return _text({"name": col.name, "status": "created_or_exists"})

    # ------------------------------------------------------------------
    if name == "chroma_add_documents":
        col_name = arguments["collection_name"]
        documents: list[str] = arguments["documents"]
        ids: list[str] = arguments["ids"]
        metadatas = arguments.get("metadatas")

        # Check for duplicates
        col = client.get_or_create_collection(col_name)
        existing = set(col.get(include=[])["ids"])
        dupes = [i for i in ids if i in existing]
        if dupes:
            raise ValueError(f"Duplicate IDs: {dupes[:5]}{'...' if len(dupes) > 5 else ''}")

        # Compute embeddings via Ollama
        embeddings = embed_texts(documents)

        col.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        return _text({"added": len(documents)})

    # ------------------------------------------------------------------
    if name == "chroma_query_documents":
        col_name = arguments["collection_name"]
        query_texts: list[str] = arguments["query_texts"]
        n_results: int = arguments.get("n_results", 5)
        where = arguments.get("where")
        where_document = arguments.get("where_document")
        include = arguments.get("include", ["documents", "metadatas", "distances"])

        col = client.get_collection(col_name)

        # Compute query embeddings via Ollama
        query_embeddings = embed_texts(query_texts)

        result = col.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
        )
        return _text(result)

    # ------------------------------------------------------------------
    if name == "chroma_get_documents":
        col_name = arguments["collection_name"]
        limit: int = arguments.get("limit", 100)
        offset: int = arguments.get("offset", 0)
        where = arguments.get("where")
        include = arguments.get("include", ["metadatas"])

        col = client.get_collection(col_name)
        result = col.get(
            limit=limit,
            offset=offset,
            where=where,
            include=include,
        )
        return _text(result)

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
