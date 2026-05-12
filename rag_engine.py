"""
The core RAG pipeline: query → retrieve → fuse → generate
answer(query, paper_id = None)  ->  dict
Returns:
        {
            "answer":   str,           # LLM-generated answer
            "sources":  list[dict],    # top fused chunks used as context
            "query":    str,
        }

summarise(paper_id, title)  ->  str
    Generate a concise summary of a single paper from its indexed chunks
"""

import json
import logging
import re
from typing import Optional
import requests
import config
from indexer import load_indexes, _embed
logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion(RRF)
def _rrf_fuse(
    faiss_results: list[dict],
    bm25_results:  list[dict],
    k:             int = config.RRF_K,
    top_n:         int = config.TOP_K,
) -> list[dict]:
    """
    Merge two ranked lists into one using Reciprocal Rank Fusion
    RRF score for a document d:
        rrf(d) = Σ  1 / (k + rank_i(d))
    where rank_i is the 1-based position in list i (unseen docs get no score)

    The fused list is sorted descending by rrf score and trimmed to top_n
    Each returned chunk gets a "_rrf_score" key added
    """
    scores: dict[str, float] = {}
    chunks: dict[str, dict]  = {}



    def _chunk_key(chunk: dict) -> str:
        return f"{chunk['paper_id']}::{chunk['chunk_idx']}"

    for rank, chunk in enumerate(faiss_results, start = 1):
        key = _chunk_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        chunks[key] = chunk

    for rank, chunk in enumerate(bm25_results, start = 1):
        key = _chunk_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        chunks.setdefault(key, chunk)

    ranked = sorted(scores.items(), key = lambda x: x[1], reverse = True)

    result = []
    for key, rrf_score in ranked[:top_n]:
        chunk = chunks[key].copy()
        chunk["_rrf_score"] = round(rrf_score, 6)
        result.append(chunk)

    return result



# Ollama
def _ollama_generate(prompt: str, system: str = "") -> str:
    """
    Call the Ollama /api/generate endpoint and return the full response text
    Streams internally but returns the complete string
    """

    url     = f"{config.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model":  config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
    }

    if system:
        payload["system"] = system

    try:
        response = requests.post(url, json = payload, stream = True, timeout = 120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL} "
            "Make sure Ollama is running: `ollama serve`"
        )

    parts = []
    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
            parts.append(chunk.get("response", ""))
            if chunk.get("done"):
                break
        except json.JSONDecodeError:
            continue

    return "".join(parts).strip()


_GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy|sup|what'?s\s+up|hiya)\W*$",
    re.IGNORECASE,
)

_GREETING_PREFIX = re.compile(
    r"^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy|sup|what'?s\s+up|hiya)[,!\s]+",
    re.IGNORECASE,
)

_RESEARCH_KEYWORDS = re.compile(
    r"\b(paper|papers|research|study|model|method|algorithm|dataset|result|finding|"
    r"experiment|approach|technique|performance|accuracy|training|inference|loss|"
    r"baseline|benchmark|architecture|network|layer|attention|transformer|embedding|"
    r"retrieval|generation|classification|evaluation|metric|abstract|conclusion|"
    r"limitation|contribution|proposed|state.of.the.art|sota)\b",
    re.IGNORECASE,
)


def _classify_query(query: str) -> tuple[str, str]:
    stripped = query.strip()

    if _GREETING_PATTERNS.match(stripped):
        return "greeting", stripped

    greeting_match = _GREETING_PREFIX.match(stripped)
    if greeting_match:
        remainder = stripped[greeting_match.end():].strip()
        if remainder:
            return "greeting_with_question", remainder

    if not _RESEARCH_KEYWORDS.search(stripped):
        return "out_of_scope", stripped

    return "research", stripped


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt"""
    sections = []
    for i, c in enumerate(chunks, start = 1):
        sections.append(
            f"[{i}] Paper: {c['title']}\n"
            f"    ID: {c['paper_id']}  Page: {c['page']}\n"
            f"    {c['text']}"
        )
    return "\n\n".join(sections)





# public API 
def answer(query:    str, paper_id: Optional[str] = None) -> dict:
    """
    Full RAG pipeline
    Parameters:
        query    : str   
        paper_id : str | None
        If given retrieval is restricted to chunks from that paper only
        (used in single-paper deep-dive mode in the UI)

    Returns:
    {
        "answer":  str,
        "sources": list[dict],   # fused top-k chunks
        "query":   str,
    }
    """

    intent, clean_query = _classify_query(query)

    if intent == "greeting":
        return {
            "answer":  "Hello 😊 I'm your research paper assistant you can ask me anything about the indexed papers questions, summaries, comparisons and more",
            "sources": [],
            "query":   query,
        }

    if intent == "out_of_scope":
        return {
            "answer":  "I'm specialized in academic research papers. I can answer questions about indexed papers, retrieve findings, compare methods or summarize studies. Please ask something related to the research content",
            "sources": [],
            "query":   query,
        }

    if intent == "greeting_with_question":
        query = clean_query

    faiss_idx, bm25_idx = load_indexes()
    if faiss_idx is None:
        return {
            "answer":  "No papers have been indexed yet Run the ingester first",
            "sources": [],
            "query":   query,
        }
    

    # retrieve 
    q_vec         = _embed([query])
    faiss_results = faiss_idx.search(q_vec, k = config.TOP_K * 2, paper_id = paper_id)
    bm25_results  = bm25_idx.search(query,  k = config.TOP_K * 2, paper_id = paper_id)

    logger.info(
        "Retrieved %d FAISS + %d BM25 chunks for query: %r",
        len(faiss_results), len(bm25_results), query[:60],
    )



    # fuse 
    fused = _rrf_fuse(faiss_results, bm25_results, top_n = config.TOP_K)
    if not fused:
        return {
            "answer":  "I couldn't find relevant chunks for that query Try rephrasing",
            "sources": [],
            "query":   query,
        }



    # generate 
    context = _build_context(fused)
    system = (
        "You are a research assistant that answers questions about academic papers. "
        "Answer using ONLY the provided context. "
        "If the context does not contain enough information, say so honestly. "
        "Be concise and precise. Cite the paper ID when referencing specific claims."
    )

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    logger.info("Calling Ollama (%s)", config.OLLAMA_MODEL)
    generated = _ollama_generate(prompt, system = system)

    return {
        "answer":  generated,
        "sources": fused,
        "query":   query,
    }



def summarise(paper_id: str, title: str) -> str:
    """
    Summarise a single paper using its indexed chunks
    Retrieves the top chunks from the paper then asks the LLM to produce
    a structured summary: objective, methods, results and limitations
    """

    faiss_idx, bm25_idx = load_indexes()
    if faiss_idx is None:
        return "No index found Run the ingester first"

    # Pull broadly from this specific paper
    q_vec  = _embed([title])
    chunks = faiss_idx.search(q_vec, k = 12, paper_id = paper_id)

    if not chunks:
        # Fallback: grab first chunks directly from BM25 with title keywords
        chunks = bm25_idx.search(title, k = 12, paper_id = paper_id)

    if not chunks:
        return f"No indexed chunks found for paper {paper_id}"

    # Sort by page order for a more coherent summary
    chunks_sorted = sorted(chunks, key = lambda c: (c["page"], c["chunk_idx"]))
    context = _build_context(chunks_sorted)

    system = (
        "You are a research assistant. Summarise the provided paper excerpts "
        "into a structured summary with these four sections:\n"
        "  • Objective – what problem does the paper address?\n"
        "  • Methods   – what approach or techniques are used?\n"
        "  • Results   – what are the key findings?\n"
        "  • Limitations – what are the acknowledged weaknesses or open questions?\n"
        "Be concise. Use bullet points within each section."
    )

    prompt = (
        f"Paper title: {title}\n"
        f"Paper ID:    {paper_id}\n\n"
        f"Excerpts:\n{context}\n\n"
        "Write the structured summary:"
    )

    logger.info("Summarising paper %s", paper_id)
    return _ollama_generate(prompt, system = system)


# Test 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("=" * 60)
    print("RAG ENGINE SMOKE TEST")
    print("=" * 60)



    # Q&A test
    # test_query = "What methods are used for efficient visual state space modeling?"
    test_query = "Hello How are You"
    print(f"\nQuery: {test_query}\n")
    result = answer(test_query)

    print("ANSWER:")
    print(result["answer"])
    print(f"\nSOURCES ({len(result['sources'])} chunks):")
    for s in result["sources"]:
        print(
            f"  [{s['paper_id']} p{s['page']}]"
            f"  rrf={s['_rrf_score']:.4f}"
            f"  {s['text'][:70].strip()!r}"
        )

    # summarise test
    faiss_idx, _ = load_indexes()
    if faiss_idx and faiss_idx.meta:
        first = faiss_idx.meta[0]
        pid, title = first["paper_id"], first["title"]
        print(f"\n{'='*60}")
        print(f"SUMMARY TEST  —  {pid}")
        print(f"Title: {title}")
        print("=" * 60)
        print(summarise(pid, title))