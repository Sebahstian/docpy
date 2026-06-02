"""ChromaDB persistent vector store wrapper for DocPy.

We bring our own embeddings (Gemini), so the collection is created with
embedding_function=None — Chroma stores/searches the vectors we give it and
never tries to embed text itself (which also avoids pulling in its default
ONNX model).
"""

from __future__ import annotations

import chromadb

from rag.types import Chunk


class VectorStore:
    """Thin wrapper over a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_path: str = "chroma_db",
        collection_name: str = "docpy",
    ) -> None:
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._make_collection()

    def _make_collection(self):
        """Create (or open) the collection we store everything in.

        embedding_function=None: we add our own vectors and never let Chroma
        embed. cosine space matches text-embedding-004's recommended metric.
        """
        return self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks + their precomputed embeddings.

        ids/embeddings/documents/metadatas are positionally aligned lists.
        Callers reset() before a fresh index so stable ids never collide.
        """
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )

    def query(self, embedding: list[float], n_results: int = 5) -> list[dict]:
        """Return the n nearest chunks as a flat list of result dicts."""
        raw = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma nests results one level per query; we sent one query → index [0].
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]
        return [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._collection.count()

    def exists(self) -> bool:
        """True if the collection already holds indexed chunks."""
        return self.count() > 0

    def reset(self) -> None:
        """Drop all data by deleting and recreating the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._make_collection()
