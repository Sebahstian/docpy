"""Tests for the loader coverage audit — uses a small stdlib module, no network."""

from __future__ import annotations

from rag.audit import AuditReport, audit_library


def test_audit_textwrap():
    # textwrap is a small, pure-Python single module (no submodule walk),
    # so this stays fast and offline.
    report = audit_library("textwrap")

    assert isinstance(report, AuditReport)
    assert report.library == "textwrap"
    assert report.total > 0
    # Pure-Python stdlib: most symbols expose signatures and source.
    assert report.with_source > 0
    assert report.with_signature > 0
    # Counts never exceed the total.
    assert report.with_docstring <= report.total
    assert sum(report.by_kind.values()) == report.total


def test_summary_renders():
    report = audit_library("textwrap")
    text = report.summary()
    assert "Audit for 'textwrap'" in text
    assert "Total symbols:" in text
