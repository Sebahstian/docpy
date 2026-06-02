"""Turn extracted Symbols into embeddable Chunks.

DocPy uses *per-symbol* chunking: each function/class/method becomes exactly
one chunk. This keeps citations precise — a retrieved chunk maps 1:1 to a
single documented symbol, so we can always point the user at the exact thing.
"""

from __future__ import annotations

from collections.abc import Iterable

from rag.types import Chunk, Symbol


class SymbolChunker:
    """Formats a Symbol into a single Chunk of text + metadata."""

    # Source code can be huge (whole classes). Truncate so we don't blow up
    # embedding token limits or stuff the prompt with thousands of lines.
    MAX_SOURCE_CHARS = 2000

    def chunk(self, symbol: Symbol) -> Chunk:
        """Build one Chunk from one Symbol."""
        return Chunk(
            id=symbol.full_id,
            text=self._format_text(symbol),
            qualname=symbol.qualname,
            module=symbol.module,
            kind=symbol.kind,
        )

    def chunk_many(self, symbols: Iterable[Symbol]) -> list[Chunk]:
        """Chunk a whole stream of Symbols into a list of Chunks."""
        return [self.chunk(s) for s in symbols]

    def _format_text(self, symbol: Symbol) -> str:
        """Build a human- and embedding-friendly text block for one symbol.

        We include whatever is available (graceful degradation, matching the
        loader): a header line, the signature, the docstring, and finally the
        (possibly truncated) source. A symbol with none of those extras still
        yields a non-empty chunk — just the header — so it stays retrievable.
        """
        parts: list[str] = []

        # Header: e.g. "method pandas.core.frame.DataFrame.merge"
        parts.append(f"{symbol.kind} {symbol.module}.{symbol.qualname}")

        if symbol.signature:
            parts.append(f"Signature: {symbol.qualname}{symbol.signature}")

        if symbol.docstring:
            parts.append(f"Description:\n{symbol.docstring}")

        if symbol.source:
            source = symbol.source
            if len(source) > self.MAX_SOURCE_CHARS:
                source = source[: self.MAX_SOURCE_CHARS] + "\n# ... (truncated)"
            parts.append(f"Source:\n{source}")

        return "\n\n".join(parts)
