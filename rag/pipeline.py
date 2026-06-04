"""RAGPipeline: ties the whole DocPy system together.

Flow:
  index_library:  loader -> chunker -> embedder -> store
  ask:            embed query -> store.query -> build prompt -> LLM -> Answer
"""

from __future__ import annotations

from collections.abc import Callable

from google import genai

from rag.chunker import SymbolChunker
from rag.embedder import GeminiEmbedder
from rag.loader import PythonDocsLoader
from rag.store import VectorStore
from rag.types import Answer, Citation


class RAGPipeline:
    """End-to-end retrieval-augmented Q&A over an installed Python library."""

    def __init__(
        self,
        api_key: str,
        persist_path: str = "chroma_db",
        chat_model: str = "gemini-2.0-flash",
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.loader = PythonDocsLoader()
        self.chunker = SymbolChunker()
        self.embedder = GeminiEmbedder(client=self.client)
        self.store = VectorStore(persist_path=persist_path)
        self.chat_model = chat_model

    def is_indexed(self) -> bool:
        """True once at least one library has been indexed into the store."""
        return self.store.exists()

    def index_library(
        self,
        library_name: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> int:
        """Extract, chunk, embed, and store every public symbol of a library.

        Returns the number of chunks indexed. Resets the store first so each
        index run starts clean (stable ids never collide on re-index).
        """

        def report(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        report(f"Loading symbols from '{library_name}'...")
        symbols = list(self.loader.load_library(library_name))

        report(f"Chunking {len(symbols)} symbols...")
        chunks = self.chunker.chunk_many(symbols)
        if not chunks:
            return 0

        report(f"Embedding {len(chunks)} chunks (this can take a while)...")
        embeddings = self.embedder.embed_documents([c.text for c in chunks])

        report("Storing in vector database...")
        self.store.reset()
        self.store.add(chunks, embeddings)

        report(f"Done — indexed {len(chunks)} symbols.")
        return len(chunks)

    def ask(self, question: str, n_results: int = 5) -> Answer:
        """Answer a question using retrieved chunks as grounding context."""
        query_vec = self.embedder.embed_query(question)
        hits = self.store.query(query_vec, n_results=n_results)

        if not hits:
            return Answer(
                text="No indexed documentation found. Index a library first.",
                citations=[],
                question=question,
            )

        prompt = self._build_prompt(question, hits)
        response = self.client.models.generate_content(
            model=self.chat_model,
            contents=prompt,
        )

        citations = [
            Citation(
                symbol_id=h["metadata"]["symbol_id"],
                qualname=h["metadata"]["qualname"],
                module=h["metadata"]["module"],
                snippet=h["document"],
            )
            for h in hits
        ]
        return Answer(text=response.text, citations=citations, question=question)

    def _build_prompt(self, question: str, hits: list[dict]) -> str:
        """Assemble a grounded prompt from retrieved chunks."""
        context_blocks = []
        for i, h in enumerate(hits, start=1):
            meta = h["metadata"]
            context_blocks.append(f"[{i}] {meta['module']}.{meta['qualname']}\n{h['document']}")
        context = "\n\n---\n\n".join(context_blocks)

        return (
            "You are a helpful assistant answering questions about a Python "
            "library, using ONLY the documentation excerpts below. If the "
            "answer isn't in the excerpts, say so honestly.\n\n"
            f"Documentation excerpts:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer clearly and concisely. Reference the relevant symbol names "
            "(e.g. module.Class.method) where helpful."
        )
