"""Gemini embeddings wrapper for DocPy.

Embeddings turn text into vectors so we can do semantic search. Gemini wants a
different `task_type` depending on whether we're embedding stored documents or a
search query — using the right one materially improves retrieval quality.
"""

from __future__ import annotations

import time

from google import genai


class GeminiEmbedder:
    """Embeds text via the Gemini embeddings API."""

    def __init__(
        self,
        client: genai.Client,
        model: str = "gemini-embedding-001",
        batch_size: int = 20,
    ) -> None:
        self.client = client
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
            result = self.client.models.embed_content(
                model=self.model,
                contents=[genai.Content(parts=[genai.Part(text=t)]) for t in batch],
                embed_config=genai.EmbedConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            vectors.extend([e.values for e in result.embeddings])
            if start + self.batch_size < len(texts):
                time.sleep(0.1)  # be gentle on the free-tier rate limit
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (task_type=retrieval_query)."""
        result = self.client.models.embed_content(
            model=self.model,
            contents=[genai.Content(parts=[genai.Part(text=text)])],
            embed_config=genai.EmbedConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values
