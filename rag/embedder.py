"""Gemini embeddings wrapper for DocPy.

Embeddings turn text into vectors so we can do semantic search. Gemini wants a
different `task_type` depending on whether we're embedding stored documents or a
search query — using the right one materially improves retrieval quality.

Uses the google-genai client SDK; the pipeline builds the genai.Client and hands
it in here. The current embedding model is `gemini-embedding-001` (the older
`embedding-001` / `text-embedding-004` models have been shut down). We request
768-dim vectors to keep the vector store compact.
"""

from __future__ import annotations

import time

from google import genai
from google.genai import types


class GeminiEmbedder:
    """Embeds text via the Gemini embeddings API."""

    def __init__(
        self,
        client: genai.Client,
        model: str = "gemini-embedding-001",
        batch_size: int = 20,
        output_dim: int = 768,
    ) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.output_dim = output_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many documents for storage (task_type=RETRIEVAL_DOCUMENT).

        Batched to stay within request limits, with a tiny sleep between
        batches as a crude guard against free-tier rate limits.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            result = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.output_dim,
                ),
            )
            # One returned embedding per input text, positionally aligned.
            vectors.extend(e.values for e in result.embeddings)
            if start + self.batch_size < len(texts):
                time.sleep(0.1)  # be gentle on the free-tier rate limit
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (task_type=RETRIEVAL_QUERY)."""
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.output_dim,
            ),
        )
        # Single input -> a single embedding.
        return result.embeddings[0].values
