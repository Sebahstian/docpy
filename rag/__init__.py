"""DocPy RAG package: RAG-powered Q&A over installed Python libraries."""

from rag.audit import AuditReport, audit_library
from rag.chunker import SymbolChunker
from rag.embedder import GeminiEmbedder
from rag.loader import PythonDocsLoader
from rag.pipeline import RAGPipeline
from rag.store import VectorStore
from rag.types import Answer, Chunk, Citation, Symbol

__all__ = [
    "Answer",
    "AuditReport",
    "Chunk",
    "Citation",
    "GeminiEmbedder",
    "PythonDocsLoader",
    "RAGPipeline",
    "Symbol",
    "SymbolChunker",
    "VectorStore",
    "audit_library",
]
