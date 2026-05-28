#Setting up the project

##Phase 0 - File Structure

docpy/
├── app.py                  # Streamlit app entry point
├── rag/
│   ├── __init__.py
│   ├── loader.py           # PythonDocsLoader using inspect
│   ├── chunker.py          # Per-symbol chunking
│   ├── embedder.py         # Gemini embeddings wrapper
│   ├── store.py            # ChromaDB wrapper
│   ├── pipeline.py         # RAGPipeline orchestrator
│   ├── audit.py
    └── types.py            
├── .streamlit/
│   ├── secrets.toml.example
│   └── secrets.toml

├── pyproject.toml          # Project metadata + dependencies
├── .python-version         # Pinned Python version
├── .gitignore
├── uv.lock                 # Locked dependencies
└── README.md

This at the time setting up the file structure and version control with Git is frustrated 
but it is a better way to understand little by little especially for beginners (like me).
Planning a project is the number one way to implement the project and keeping it steady witout jumping files back and forth of the project.
