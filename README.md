# DocPy

RAG-powered Q&A for Python library documentation. Ask natural-language questions about any installed Python library and get answers with citations to specific functions and classes.

🚧 **Status:** In active development (Phase 0 complete)

## Stack

- **Gemini API** — LLM + embeddings
- **ChromaDB** — local vector store
- **Streamlit** — web UI
- **Python `inspect`** — extracts symbols from installed libraries

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/YOUR_USERNAME/docpy.git
cd docpy
uv sync
uv run streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com/apikey, then copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your key.