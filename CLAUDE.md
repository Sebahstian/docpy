# DocPy

RAG-powered Q&A for Python library documentation. Users index any installed Python library and ask natural-language questions about it. Answers are grounded in retrieved docstrings and source code, with citations.

## Stack

- **Frontend**: Streamlit
- **LLM**: Gemini 2.0 Flash (`google-generativeai`)
- **Embeddings**: Gemini `models/embedding-001`
- **Vector DB**: ChromaDB (persistent, cosine distance, bring-your-own embeddings)
- **Python**: ≥3.11, managed with `uv`
- **Linting**: Ruff (`line-length = 100`, rules E/F/I/W/UP)

## Running the app

```bash
uv run streamlit run app.py
```

Requires `.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`):
```toml
GEMINI_API_KEY = "your-key-from-aistudio.google.com"
```

## Development

```bash
uv run pytest          # run tests
uv run ruff check .    # lint
uv run ruff format .   # format
```

## Folder / file structure

```
docpy/
├── app.py                        # Streamlit UI — sidebar index + chat interface
├── main.py                       # CLI entry point (thin wrapper)
│
├── rag/                          # Core RAG pipeline (pure Python, no Streamlit)
│   ├── types.py                  # Data models: Symbol, Chunk, Citation, Answer
│   ├── loader.py                 # PythonDocsLoader — introspects installed libraries via `inspect`
│   ├── chunker.py                # SymbolChunker — 1 Symbol → 1 Chunk (precise citations)
│   ├── embedder.py               # GeminiEmbedder — batched embed_documents / embed_query
│   ├── store.py                  # VectorStore — ChromaDB persistent collection wrapper
│   ├── pipeline.py               # RAGPipeline — orchestrates index_library() and ask()
│   ├── audit.py                  # Audit/debug helpers
│   └── __init__.py
│
├── tests/
│   ├── test_chunker.py
│   ├── test_pipeline.py
│   ├── test_audit.py
│   └── __init__.py
│
├── .streamlit/
│   └── secrets.toml.example      # Template for GEMINI_API_KEY
│
├── pyproject.toml                # Project metadata, deps, ruff config
├── requirements.txt              # Streamlit Cloud mirror of pyproject.toml deps
├── uv.lock                       # Locked dependency tree
├── .python-version               # Python version pin for uv
├── README.md
└── PROJECT_JOURNEY.md            # Dev log / design decisions
```

## Data flow

```
index_library:
  PythonDocsLoader  →  SymbolChunker  →  GeminiEmbedder  →  VectorStore
  (inspect module)     (1 symbol/chunk)  (batch embed)       (ChromaDB upsert)

ask(question):
  GeminiEmbedder(query)  →  VectorStore.query()  →  build_prompt()  →  Gemini LLM  →  Answer
```

## Key design decisions

- **Per-symbol chunking**: each function/class/method is exactly one chunk, so retrieved citations map 1:1 to a symbol — no ambiguous sub-chunk references.
- **Bring-your-own embeddings**: `embedding_function=None` in ChromaDB; Gemini embeddings are passed in directly, avoiding Chroma's default ONNX model download.
- **Graceful degradation**: C-extension symbols (numpy, scipy) often lack signatures/source — the loader catches all failures and returns whatever is available (usually just the docstring).
- **Store reset on re-index**: `store.reset()` before every `index_library` run prevents duplicate/stale chunk collisions since chunk ids are stable (`module.qualname`).
