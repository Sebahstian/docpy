"""Offline wiring test for RAGPipeline.

We monkeypatch every Gemini network call (configure, embed_content, the chat
model) and point ChromaDB at a tmp dir, so the whole index -> ask flow runs
without an API key or network access.
"""

from __future__ import annotations

import google.generativeai as genai
import pytest

from rag.pipeline import RAGPipeline
from rag.types import Symbol


class _FakeResponse:
    text = "requests.get sends a GET request."


class _FakeModel:
    def __init__(self, *args, **kwargs):
        pass

    def generate_content(self, prompt):
        return _FakeResponse()


def _fake_embed_content(model, content, task_type=None):
    """Deterministic 3-dim vectors; handles both batch and single inputs."""

    def vec(text: str) -> list[float]:
        return [float(len(text)), float(text.count("e")), 1.0]

    if isinstance(content, list):
        return {"embedding": [vec(t) for t in content]}
    return {"embedding": vec(content)}


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(genai, "configure", lambda **kwargs: None)
    monkeypatch.setattr(genai, "embed_content", _fake_embed_content)
    monkeypatch.setattr(genai, "GenerativeModel", _FakeModel)

    pipe = RAGPipeline(api_key="fake-key", persist_path=str(tmp_path / "chroma"))

    # Stub the loader so we don't depend on any installed third-party library.
    def fake_load(library_name):
        return iter(
            [
                Symbol(
                    name="get",
                    qualname="get",
                    module="requests.api",
                    kind="function",
                    signature="(url, params=None)",
                    docstring="Send a GET request.",
                    source="def get(url, params=None): ...",
                ),
                Symbol(
                    name="post",
                    qualname="post",
                    module="requests.api",
                    kind="function",
                    signature="(url, data=None)",
                    docstring="Send a POST request.",
                    source="def post(url, data=None): ...",
                ),
            ]
        )

    monkeypatch.setattr(pipe.loader, "load_library", fake_load)
    return pipe


def test_index_and_ask(pipeline):
    assert pipeline.is_indexed() is False

    count = pipeline.index_library("requests")
    assert count == 2
    assert pipeline.is_indexed() is True

    answer = pipeline.ask("How do I send a GET request?", n_results=2)
    assert answer.text == "requests.get sends a GET request."
    assert answer.question == "How do I send a GET request?"
    # One citation per retrieved hit, carrying real metadata.
    assert len(answer.citations) == 2
    assert {c.symbol_id for c in answer.citations} == {
        "requests.api.get",
        "requests.api.post",
    }


def test_reindex_replaces_contents(pipeline):
    pipeline.index_library("requests")
    # Re-indexing resets the store rather than duplicating ids.
    pipeline.index_library("requests")
    assert pipeline.store.count() == 2
