"""
Streamlit front-end for the Research Paper Assistant
Two modes toggled from the sidebar:
    Ask Anything   –  RAG Q&A across all indexed papers
    Deep Dive      –  pick one paper, ask questions or get a summary
"""

import streamlit as st
import config
from ingest     import run as run_ingest
from indexer    import load_indexes
from rag_engine import answer, summarise
st.set_page_config(page_title = config.APP_TITLE, page_icon = config.APP_ICON, layout = "wide")


def _all_papers() -> list[dict]:
    faiss_idx, _ = load_indexes()
    if faiss_idx is None or not faiss_idx.meta:
        return []
    seen, papers = set(), []
    for chunk in faiss_idx.meta:
        pid = chunk["paper_id"]
        if pid not in seen:
            seen.add(pid)
            papers.append({"paper_id": pid, "title": chunk["title"]})
    return papers


def _ingest_button() -> None:
    if st.button("Fetch & Ingest New Papers", use_container_width=True):
        with st.spinner("Fetching and indexing papers …"):
            try:
                run_ingest()
                st.success("Ingestion complete!")
                st.rerun()
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources  ({len(sources)} chunks)"):
        for i, s in enumerate(sources, 1):
            st.markdown(
                f"**[{i}]** `{s['paper_id']}` — *{s['title']}*  "
                f"· page {s['page']}  · RRF {s['_rrf_score']:.4f}"
            )
            st.caption(s["text"][:300] + ("…" if len(s["text"]) > 300 else ""))
            if i < len(sources):
                st.divider()


def _chat_history_key(mode: str, paper_id: str = "") -> str:
    return f"history_{mode}_{paper_id}"


def _render_chat(mode: str, paper_id: str = "", paper_title: str = "") -> None:
    history_key = _chat_history_key(mode, paper_id)
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    query = st.chat_input("Ask a question about the papers …")
    if not query:
        return

    with st.chat_message("user"):
        st.markdown(query)
    st.session_state[history_key].append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            result = answer(query, paper_id=paper_id if mode == "deep_dive" else None)

        st.markdown(result["answer"])
        _render_sources(result["sources"])

    st.session_state[history_key].append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })


def main() -> None:
    st.sidebar.title(config.APP_ICON + "  " + config.APP_TITLE)
    st.sidebar.divider()

    mode = st.sidebar.radio("Mode", ["Ask Anything", "Deep Dive"], label_visibility="collapsed")

    st.sidebar.divider()
    _ingest_button()

    papers = _all_papers()
    st.sidebar.caption(f"{len(papers)} paper(s) indexed")

    st.sidebar.divider()
    if st.sidebar.button("Clear chat history", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("history_"):
                del st.session_state[key]
        st.rerun()

    if mode == "Ask Anything":
        st.title("Ask Anything")
        if not papers:
            st.info("No papers indexed yet Use the sidebar button to fetch and ingest papers")
            return
        _render_chat(mode="ask_anything")

    else:
        st.title("Deep Dive")
        if not papers:
            st.info("No papers indexed yet Use the sidebar button to fetch and ingest papers")
            return

        options  = {p["title"]: p for p in papers}
        selected = st.selectbox("Select a paper", list(options.keys()))
        paper    = options[selected]

        st.caption(f"`{paper['paper_id']}`")
        st.divider()

        col1, col2 = st.columns([1, 5])
        with col1:
            summarise_btn = st.button("Summarise", use_container_width=True)
        with col2:
            clear_summary = st.button("Clear summary", use_container_width=True)

        summary_key = f"summary_{paper['paper_id']}"
        if clear_summary:
            st.session_state.pop(summary_key, None)

        if summarise_btn:
            with st.spinner("Summarising …"):
                summary = summarise(paper["paper_id"], paper["title"])
            st.session_state[summary_key] = summary

        if summary_key in st.session_state:
            st.markdown(st.session_state[summary_key])
            st.divider()

        _render_chat(mode="deep_dive", paper_id=paper["paper_id"], paper_title=paper["title"])


if __name__ == "__main__":
    main()


