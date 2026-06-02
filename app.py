"""DocPy Streamlit app: ask natural-language questions about a Python library.

Run with:  uv run streamlit run app.py
Requires .streamlit/secrets.toml with GEMINI_API_KEY.
"""

from __future__ import annotations

import streamlit as st

from rag.pipeline import RAGPipeline
from rag.types import Answer

st.set_page_config(page_title="DocPy", page_icon="📚")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    """Create the pipeline once and reuse it across reruns."""
    api_key = st.secrets["GEMINI_API_KEY"]
    return RAGPipeline(api_key=api_key)


def render_answer(answer: Answer) -> None:
    """Render an Answer's text plus a collapsible list of its sources."""
    st.markdown(answer.text)
    if answer.citations:
        with st.expander(f"Sources ({len(answer.citations)})"):
            for c in answer.citations:
                st.markdown(f"**{c.module}.{c.qualname}**")
                st.code(c.snippet)


def main() -> None:
    st.title("DocPy 📚")
    st.caption("RAG-powered Q&A for Python library documentation")

    pipeline = get_pipeline()

    # --- Sidebar: index a library -------------------------------------
    with st.sidebar:
        st.header("Index a library")
        library_name = st.text_input("Library name", value="requests")
        if st.button("Index", type="primary"):
            status = st.empty()
            with st.spinner(f"Indexing {library_name}..."):
                try:
                    count = pipeline.index_library(
                        library_name,
                        progress_callback=status.write,
                    )
                    st.session_state["indexed_library"] = library_name
                    st.success(f"Indexed {count} symbols from {library_name}.")
                except ImportError:
                    st.error(f"'{library_name}' is not installed in this environment.")
                except Exception as exc:  # surface API/quota errors to the user
                    st.error(f"Indexing failed: {exc}")

        if pipeline.is_indexed():
            st.info(f"Ready — {pipeline.store.count()} symbols indexed.")

    # --- Main area: chat ----------------------------------------------
    if not pipeline.is_indexed():
        st.info("Index a library from the sidebar to get started.")
        return

    if "history" not in st.session_state:
        st.session_state["history"] = []  # list[Answer]

    # Replay prior Q&A
    for answer in st.session_state["history"]:
        with st.chat_message("user"):
            st.markdown(answer.question)
        with st.chat_message("assistant"):
            render_answer(answer)

    question = st.chat_input("Ask a question about the indexed library...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = pipeline.ask(question)
            render_answer(answer)
        st.session_state["history"].append(answer)


if __name__ == "__main__":
    main()
