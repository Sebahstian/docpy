"""Offline wiring test for RAGPipeline.

We monkeypatch every Gemini network call (Client creation, embed_content, the
chat model) and point ChromaDB at a tmp dir, so the whole index -> ask flow
runs without an API key or network access.
"""

from __future__ import annotations

from google import genai
import pytest

from rag.pipeline import RAGPipeline
from rag.types import Symbol


class _FakePart:
    def __init__(self, text: str = ""):
        self.text = text


class _FakeContent:
    def __init__(self, parts=None):
        self.parts = parts or []


class _FakeEmbedding:
    values: list[float]

    def __init__(self, values: list[float]):
        self.values = values


class _FakeEmbedResponse:
    embeddings: list[_FakeEmbedding]

    def __init__(self, embeddings: list[_FakeEmbedding]):
        self.embeddings = embeddings


class _FakeResponse:
    text = "requests.get sends a GET request."


class _FakeModels:
    def __init__(self, *args, **kwargs):
        pass

    def embed_content(self, model, contents, embed_config=None):
        """Deterministic 3-dim vectors from text length and char count."""

        def vec(text: str) -> list[float]:
            return [float(len(text)), float(text.count("e")), 1.0]

        # Contents is a list of genai.Content objects; extract text from parts
        embeddings = []
        for content in contents:
            text = content.parts[0].text if content.parts else ""
            embeddings.append(_FakeEmbedding(vec(text)))
        return _FakeEmbedResponse(embeddings)

    def generate_content(self, model, contents):
        return _FakeResponse()


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.models = _FakeModels()


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(genai, "Client", _FakeClient)
    monkeypatch.setattr(genai, "Content", _FakeContent)
    monkeypatch.setattr(genai, "Part", _FakePart)
    monkeypatch.setattr(genai, "EmbedConfig", lambda task_type: None)

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
