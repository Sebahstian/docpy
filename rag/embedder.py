"""Local ONNX embeddings wrapper for DocPy.

Uses ChromaDB's bundled DefaultEmbeddingFunction (all-MiniLM-L6-v2 via ONNX)
so indexing requires no API key and hits no rate limits. The first call
downloads the ~23 MB ONNX model to a local cache; subsequent calls are instant.

Produces 384-dim vectors. The Gemini API is only used for generating answers.
"""

from __future__ import annotations

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class LocalEmbedder:
    """Embeds text locally via the all-MiniLM-L6-v2 ONNX model."""

    def __init__(self) -> None:
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many documents for storage."""
        return list(self._ef(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return self._ef([text])[0]
