"""
Builds and persists the two search indexes used by the retriever:

  1. FAISS flat index  – dense semantic search via sentence-transformers
  2. BM25 index        – sparse keyword search via rank_bm25


add_chunks(chunks: list[dict]) -> None
    Embed chunks, add them to both indexes and persist to CACHE_DIR

load_indexes() -> tuple[FAISSIndex, BM25Index] | tuple[None, None]
    Load persisted indexes from CACHE_DIR and returns (None, None) if not found
"""


import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import config
logger = logging.getLogger(__name__)


# cache paths
_FAISS_INDEX_PATH  = config.CACHE_DIR / "faiss.index"
_FAISS_META_PATH   = config.CACHE_DIR / "faiss_meta.pkl"    
_BM25_PATH         = config.CACHE_DIR / "bm25.pkl"


# embedding model
_embedder_model = None
def _get_embedder():
    global _embedder_model
    if _embedder_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model '%s'", config.EMBEDDING_MODEL)
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder



def _embed(texts: list[str]) -> np.ndarray:
    """Return a float32 numpy array of shape (len(texts), dim)"""
    model = _get_embedder()
    vecs = model.encode(texts, show_progress_bar = False, convert_to_numpy = True)
    return vecs.astype(np.float32)




# FAISS index 
@dataclass
class FAISSIndex:
    index: object          
    meta:  list[dict] = field(default_factory = list)   

    def save(self) -> None:
        import faiss
        faiss.write_index(self.index, str(_FAISS_INDEX_PATH))
        with open(_FAISS_META_PATH, "wb") as f:
            pickle.dump(self.meta, f)
        logger.info("FAISS index saved (%d vectors)", self.index.ntotal)



    @classmethod
    def load(cls) -> Optional["FAISSIndex"]:
        if not _FAISS_INDEX_PATH.exists() or not _FAISS_META_PATH.exists():
            return None
        import faiss
        index = faiss.read_index(str(_FAISS_INDEX_PATH))
        with open(_FAISS_META_PATH, "rb") as f:
            meta = pickle.load(f)
        logger.info("FAISS index loaded (%d vectors)", index.ntotal)
        return cls(index=index, meta=meta)



    def add(self, vecs: np.ndarray, chunks: list[dict]) -> None:
        import faiss
        # Normalize for cosine similarity (IndexFlatIP on unit vectors = cosine)
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.meta.extend(chunks)



    def search(
        self,
        query_vec: np.ndarray,
        k: int = config.TOP_K,
        paper_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Return up to k chunk dicts
        If paper_id is given only chunks from that paper are returned
        Results below SCORE_THRESHOLD are discarded
        """
        import faiss
        if self.index.ntotal == 0:
            return []

        q = query_vec.copy().astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)

        # Over-fetch so we can filter by paper_id and threshold
        fetch_k = min(self.index.ntotal, k * 10 if paper_id else k * 2)
        scores, indices = self.index.search(q, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score < config.SCORE_THRESHOLD:
                continue
            chunk = self.meta[idx].copy()
            chunk["_faiss_score"] = float(score)
            if paper_id and chunk.get("paper_id") != paper_id:
                continue
            results.append(chunk)
            if len(results) >= k:
                break
        return results


# BM25 index
@dataclass
class BM25Index:
    bm25:   object              # BM25Okapi instance
    tokens: list[list[str]]    
    chunks: list[dict]

    def save(self) -> None:
        with open(_BM25_PATH, "wb") as f:
            pickle.dump((self.bm25, self.tokens, self.chunks), f)
        logger.info("BM25 index saved (%d documents)", len(self.chunks))



    @classmethod
    def load(cls) -> Optional["BM25Index"]:
        if not _BM25_PATH.exists():
            return None
        with open(_BM25_PATH, "rb") as f:
            bm25, tokens, chunks = pickle.load(f)
        logger.info("BM25 index loaded (%d documents)", len(chunks))
        return cls(bm25 = bm25, tokens = tokens, chunks = chunks)



    @classmethod
    def build(cls, all_tokens: list[list[str]], all_chunks: list[dict]) -> "BM25Index":
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(all_tokens)
        return cls(bm25 = bm25, tokens = all_tokens, chunks = all_chunks)



    def search(
        self,
        query: str,
        k: int = config.TOP_K,
        paper_id: Optional[str] = None,
    ) -> list[dict]:
        """Return up to k chunk dicts ranked by BM25 score"""
        if not self.chunks:
            return []

        q_tokens = _tokenize(query)
        scores   = self.bm25.get_scores(q_tokens)
        ranked = sorted(
            ((score, i) for i, score in enumerate(scores) if score > 0),   # Pair each chunk with its score, filter, sort
            reverse=True,
        )

        results = []
        for score, i in ranked:
            chunk = self.chunks[i].copy()
            if paper_id and chunk.get("paper_id") != paper_id:
                continue
            chunk["_bm25_score"] = float(score)
            results.append(chunk)
            if len(results) >= k:
                break
        return results
    

# Tokenizer 
def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric and drop empty tokens"""
    import re
    return [t for t in re.split(r"\W+", text.lower()) if t]


# Public API
def add_chunks(chunks: list[dict]) -> None:
    """
    Embed chunks and add them to the persisted FAISS + BM25 indexes
    Existing index data is preserved this function is additive
    Parameters
    ----------
    chunks : list[dict]
        Chunk dicts as produced by parser.parse_paper()
    """

    if not chunks:
        logger.warning("add_chunks called with empty list so nothing to do")
        return

    import faiss

    # FAISS 
    faiss_idx = FAISSIndex.load()
    if faiss_idx is None:
        dim = _get_embedder().get_embedding_dimension()
        faiss_idx = FAISSIndex(index = faiss.IndexFlatIP(dim))
        logger.info("Created new FAISS index (dim = %d)", dim)

    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d chunk(s)", len(texts))
    vecs = _embed(texts)
    faiss_idx.add(vecs, chunks)
    faiss_idx.save()




    # BM25 (BM25Okapi must be rebuilt from scratch when new docs are added)
    bm25_idx = BM25Index.load()

    if bm25_idx is not None:
        all_tokens = bm25_idx.tokens + [_tokenize(c["text"]) for c in chunks]
        all_chunks = bm25_idx.chunks + chunks

    else:
        all_tokens = [_tokenize(c["text"]) for c in chunks]
        all_chunks = list(chunks)

    new_bm25 = BM25Index.build(all_tokens, all_chunks)
    new_bm25.save()

    logger.info(
        "Indexes updated FAISS: %d vectors | BM25: %d documents",
        faiss_idx.index.ntotal,
        len(new_bm25.chunks),
    )



def load_indexes() -> tuple:
    """
    Load both indexes from CACHE_DIR
    Returns:
    (FAISSIndex, BM25Index)  if both exist
    (None, None)             if neither has been built yet
    """
    faiss_idx = FAISSIndex.load()
    bm25_idx  = BM25Index.load()

    if faiss_idx is None or bm25_idx is None:
        logger.warning("One or both indexes not found run the ingester first")
        return None, None

    return faiss_idx, bm25_idx




# Test
if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "%(levelname)s  %(message)s")
    from fetcher import fetch_new_papers
    from parser  import parse_paper


    papers = fetch_new_papers()
    if not papers:
        print("No new papers so nothing to index")
    else:
        test_paper = papers[0]
        print(f"\nIndexing: [{test_paper['paper_id']}] {test_paper['title'][:60]}")
        chunks = parse_paper(test_paper)
        add_chunks(chunks)

        print("\n── Search smoke-test ──────────────────────────────────────")
        faiss_idx, bm25_idx = load_indexes()

        query = "attention mechanism transformer"
        print(f"Query: '{query}'\n")
        q_vec = _embed([query])


        faiss_results = faiss_idx.search(q_vec, k = 3)
        print(f"FAISS top-{len(faiss_results)}:")
        for r in faiss_results:
            print(f"  [{r['paper_id']} p{r['page']}] score = {r['_faiss_score']:.3f}  {r['text'][:80].strip()!r}")


        bm25_results = bm25_idx.search(query, k = 3)
        print(f"\nBM25  top-{len(bm25_results)}:")
        for r in bm25_results:
            print(f"  [{r['paper_id']} p{r['page']}] score = {r['_bm25_score']:.3f}  {r['text'][:80].strip()!r}")
