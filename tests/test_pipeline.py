"""Offline wiring test for RAGPipeline.

Embeddings use a monkeypatched DefaultEmbeddingFunction (no ONNX download).
The Gemini chat client is also faked so the whole index -> ask flow runs
without any API key or network access.
"""

from __future__ import annotations

import pytest
from google import genai

import rag.embedder as embedder_module
from rag.pipeline import RAGPipeline
from rag.types import Symbol


def _vec(text: str) -> list[float]:
    return [float(len(text)), float(text.count("e")), 1.0]


class _FakeEmbeddingFunction:
    """Deterministic 3-dim vectors; no ONNX model required."""

    def __call__(self, texts):
        return [_vec(t) for t in texts]


class _FakeResponse:
    text = "requests.get sends a GET request."


class _FakeModels:
    @staticmethod
    def generate_content(model, contents):
        return _FakeResponse()


class _FakeClient:
    models = _FakeModels()


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    # Patch the local embedder so no ONNX model is downloaded.
    monkeypatch.setattr(embedder_module, "DefaultEmbeddingFunction", _FakeEmbeddingFunction)
    # Patch the Gemini client used for chat generation.
    monkeypatch.setattr(genai, "Client", lambda **kwargs: _FakeClient())

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
