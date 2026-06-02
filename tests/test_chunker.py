"""Tests for SymbolChunker — pure, no network or external services."""

from __future__ import annotations

from rag.chunker import SymbolChunker
from rag.types import Symbol


def _symbol(**overrides) -> Symbol:
    defaults = dict(
        name="merge",
        qualname="DataFrame.merge",
        module="pandas.core.frame",
        kind="method",
        signature="(self, right, how='inner')",
        docstring="Merge DataFrame objects.",
        source="def merge(self, right, how='inner'):\n    return ...",
    )
    defaults.update(overrides)
    return Symbol(**defaults)


def test_chunk_includes_all_sections():
    chunk = SymbolChunker().chunk(_symbol())

    assert chunk.id == "pandas.core.frame.DataFrame.merge"
    assert chunk.kind == "method"
    # Header + each labelled section is present.
    assert "method pandas.core.frame.DataFrame.merge" in chunk.text
    assert "Signature: DataFrame.merge(self, right, how='inner')" in chunk.text
    assert "Description:\nMerge DataFrame objects." in chunk.text
    assert "Source:\n" in chunk.text


def test_source_is_truncated():
    chunker = SymbolChunker()
    long_source = "x" * (chunker.MAX_SOURCE_CHARS + 500)
    chunk = chunker.chunk(_symbol(source=long_source))

    assert "# ... (truncated)" in chunk.text
    # Original full-length source must not survive intact.
    assert long_source not in chunk.text


def test_bare_symbol_still_yields_header():
    chunk = SymbolChunker().chunk(_symbol(signature=None, docstring=None, source=None))

    assert chunk.text == "method pandas.core.frame.DataFrame.merge"
    assert chunk.text  # non-empty so it stays retrievable


def test_chunk_many():
    chunks = SymbolChunker().chunk_many([_symbol(), _symbol(qualname="DataFrame.join")])
    assert len(chunks) == 2
    assert chunks[1].qualname == "DataFrame.join"


def test_metadata_is_flat_strings():
    meta = SymbolChunker().chunk(_symbol()).to_metadata()
    assert meta == {
        "symbol_id": "pandas.core.frame.DataFrame.merge",
        "qualname": "DataFrame.merge",
        "module": "pandas.core.frame",
        "kind": "method",
    }
    assert all(isinstance(v, str) for v in meta.values())
