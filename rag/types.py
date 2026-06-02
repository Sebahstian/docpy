"""Core data types for the DocPy RAG pipeline.

These dataclasses are the "shape" of data moving through the system:
- Symbol: one extracted item from a Python library (function, class, method)
- Citation: where a piece of retrieved info came from
- Answer: the final response with text + sources
"""

from dataclasses import dataclass, field


@dataclass
class Symbol:
    """A single extractable item from a Python library.

    Attributes:
        name: Short name, e.g. "merge"
        qualname: Fully qualified name, e.g. "DataFrame.merge"
        module: Module path, e.g. "pandas.core.frame"
        kind: One of "function", "class", "method"
        signature: Function/method signature as a string, or None if unavailable
            (C-extension symbols often have no inspectable signature)
        docstring: The cleaned docstring, or None if missing
        source: Source code as a string, or None if unavailable
            (C-extension symbols have no Python source)
    """

    name: str
    qualname: str
    module: str
    kind: str
    signature: str | None = None
    docstring: str | None = None
    source: str | None = None

    @property
    def full_id(self) -> str:
        """Unique identifier across all libraries, e.g. 'pandas.core.frame.DataFrame.merge'."""
        return f"{self.module}.{self.qualname}"


@dataclass
class Chunk:
    """One embeddable unit of text derived from a Symbol.

    A Chunk is what actually gets stored in the vector DB: a stable id, the
    text we embed (and later show to the LLM), and flat metadata we use to
    rebuild a Citation after retrieval.

    Attributes:
        id: Stable unique id, equal to Symbol.full_id
        text: The formatted, embeddable document text
        qualname: Carried through for Citation display, e.g. "DataFrame.merge"
        module: Carried through for Citation display
        kind: One of "function", "class", "method"
    """

    id: str
    text: str
    qualname: str
    module: str
    kind: str

    def to_metadata(self) -> dict[str, str]:
        """Flat metadata dict for ChromaDB (no None / nested values allowed)."""
        return {
            "symbol_id": self.id,
            "qualname": self.qualname,
            "module": self.module,
            "kind": self.kind,
        }


@dataclass
class Citation:
    """A reference to a Symbol that supported part of an answer."""

    symbol_id: str  # The full_id of the Symbol
    qualname: str  # Display name, e.g. "DataFrame.merge"
    module: str  # Module path
    snippet: str  # The chunk text that was retrieved


@dataclass
class Answer:
    """The result of asking the RAG pipeline a question."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    question: str = ""
