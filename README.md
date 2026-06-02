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

## Deploy a public link (Streamlit Community Cloud)

This gives you a URL like `https://docpy-yourname.streamlit.app` you can open on
any phone or share — no local server needed.

1. Push this repo to GitHub (already done).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Set:
   - **Repository:** `Sebahstian/docpy`
   - **Branch:** the branch you want to deploy
   - **Main file path:** `app.py`
5. Open **Advanced settings** → **Secrets** and paste your key in TOML form:
   ```toml
   GEMINI_API_KEY = "your-real-key"
   ```
6. Click **Deploy**. First build takes a few minutes (it installs
   `requirements.txt`).

Once it's live, index a library from the sidebar (`requests` is preloaded) and
ask away. Note: you can only query libraries that are installed on the
server — they must appear in `requirements.txt`.