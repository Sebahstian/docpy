"""Gemini embeddings wrapper for DocPy.

Embeddings turn text into vectors so we can do semantic search. Gemini wants a
different `task_type` depending on whether we're embedding stored documents or a
search query — using the right one materially improves retrieval quality.

Assumes genai.configure(api_key=...) was already called (the pipeline does it).
"""

from __future__ import annotations

import time

import google.generativeai as genai


class GeminiEmbedder:
    """Embeds text via the Gemini embeddings API."""

    def __init__(
        self,
        model: str = "models/embedding-001",
        batch_size: int = 100,
    ) -> None:
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many documents for storage (task_type=retrieval_document).

        Batched to stay within request limits, with a tiny sleep between
        batches as a crude guard against free-tier rate limits.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            result = genai.embed_content(
                model=self.model,
                content=batch,
                task_type="retrieval_document",
            )
            # For a list input, result["embedding"] is a list of vectors.
            vectors.extend(result["embedding"])
            if start + self.batch_size < len(texts):
                time.sleep(0.1)  # be gentle on the free-tier rate limit
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (task_type=retrieval_query)."""
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query",
        )
        # For a single string input, result["embedding"] is one vector.
        return result["embedding"]
