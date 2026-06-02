"""Coverage audit for the DocPy loader.

Runs the loader over a library and reports how complete the extraction is: how
many symbols we found, and what fraction expose a signature, docstring, or
source. Helps explain why some libraries (C extensions) answer worse than
pure-Python ones — they simply give `inspect` far less to work with.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from rag.loader import PythonDocsLoader


@dataclass
class AuditReport:
    """Extraction-quality statistics for one library."""

    library: str
    total: int = 0
    with_signature: int = 0
    with_docstring: int = 0
    with_source: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def _pct(self, n: int) -> float:
        return (100.0 * n / self.total) if self.total else 0.0

    def summary(self) -> str:
        """A human-readable multi-line summary."""
        kinds = ", ".join(f"{k}: {v}" for k, v in sorted(self.by_kind.items()))
        return (
            f"Audit for '{self.library}'\n"
            f"  Total symbols: {self.total}\n"
            f"  With signature: {self.with_signature} ({self._pct(self.with_signature):.1f}%)\n"
            f"  With docstring: {self.with_docstring} ({self._pct(self.with_docstring):.1f}%)\n"
            f"  With source:    {self.with_source} ({self._pct(self.with_source):.1f}%)\n"
            f"  By kind: {kinds}"
        )


def audit_library(library_name: str) -> AuditReport:
    """Load a library and tally extraction coverage stats."""
    loader = PythonDocsLoader()
    report = AuditReport(library=library_name)
    kinds: Counter[str] = Counter()

    for symbol in loader.load_library(library_name):
        report.total += 1
        kinds[symbol.kind] += 1
        if symbol.signature:
            report.with_signature += 1
        if symbol.docstring:
            report.with_docstring += 1
        if symbol.source:
            report.with_source += 1

    report.by_kind = dict(kinds)
    return report
