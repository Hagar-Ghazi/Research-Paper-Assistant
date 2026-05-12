"""
Downloads a paper's PDF and turns it into clean text chunks
parse_paper(paper: dict) -> list[dict]
    Downloads the PDF (if not cached) extracts + cleans text with PyMuPDF
    then splits it into overlapping chunks

    Each chunk dict contains:
        paper_id  : str   – from the input paper dict
        title     : str   – from the input paper dict
        chunk_idx : int   – 0-based position within the paper
        text      : str   – the chunk text
        page      : int   – PDF page number where the chunk starts (1-based)
"""

import re
import logging
from pathlib import Path
import config
try:
    import fitz  
except ImportError:
    import pymupdf as fitz  



logger = logging.getLogger(__name__)


# PDF Download
def _pdf_path(paper_id: str) -> Path:
    """Return the local cache path for a paper's PDF"""
    config.PDF_DIR.mkdir(parents = True, exist_ok = True)
    return config.PDF_DIR / f"{paper_id}.pdf"




def _download_pdf(pdf_url: str, dest: Path) -> None:
    """Stream-download a PDF to *dest*"""
    import urllib.request
    import time

    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(pdf_url, timeout = 60) as resp:
                dest.write_bytes(resp.read())
            logger.info("Downloaded PDF --> %s", dest.name)
            return
        
        except Exception as exc:
            logger.warning("PDF download failed (attempt %d/3): %s", attempt, exc)
            if attempt < 3:
                time.sleep(5)
    raise RuntimeError(f"Could not download PDF from {pdf_url}")


# text extraction
def _extract_text(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Extract text page-by-page using PyMuPDF
    Returns a list of (page_number, page_text) tuples (1-based page numbers)
    """
    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start = 1):
            text = page.get_text("text")
            if text.strip():
                pages.append((page_num, text))
    return pages


# text cleaning
def _clean(text: str) -> str:
    """
    Light cleaning pass that removes common arXiv PDF artefacts while
    preserving paragraph structure
    """
    
    text = re.sub(r"-\n(\w)", r"\1", text)         # Remove hyphenation at line breaks: "computa-\ntion" → "computation"
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)   # Collapse mid-paragraph line breaks (not double newlines = paragraph breaks)
    text = re.sub(r"[ \t]{2,}", " ", text)         # Collapse runs of spaces/tabs

    lines = []
    for line in text.splitlines():                 # Drop lines that look like page headers/footers
        stripped = line.strip()
        if re.fullmatch(r"[\d\s\-–—|]+", stripped) and len(stripped) < 10:
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)         # Normalise paragraph breaks to exactly two newlines
    return text.strip()



# chunking
def _chunk_text(
    pages: list[tuple[int, str]],
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int    = config.CHUNK_OVERLAP,
) -> list[tuple[int, str]]:
    """
    Recursive character-level chunker
    Strategy (highest → lowest priority split points):
        1. Double newline  (paragraph boundary)
        2. Single newline
        3. Period + space  (sentence boundary)
        4. Space           (word boundary)

    Returns list of (page_number, chunk_text) tuples
    The page number recorded is where the chunk started
    """

    SEPARATORS = ["\n\n", "\n", ". ", " "]
    stream: list[tuple[int, str]] = []
    for page_num, text in pages:
        for ch in _clean(text):
            stream.append((page_num, ch))




    def _split(chars: list[tuple[int, str]], sep: str) -> list[list[tuple[int, str]]]:
        """Split chars on every occurrence of sep (multi-char aware)"""
        result, current = [], []
        i = 0
        sep_chars = list(sep)
        sep_len   = len(sep_chars)
        while i < len(chars):
            window = [c for _, c in chars[i : i + sep_len]]
            if window == sep_chars:
                if current:
                    result.append(current)
                current = []
                i += sep_len
            else:
                current.append(chars[i])
                i += 1
        if current:
            result.append(current)
        return result




    def _recursive_chunk(chars: list[tuple[int, str]], depth: int = 0) -> list[list[tuple[int, str]]]:
        if len(chars) <= chunk_size:
            return [chars] if chars else []

        sep = SEPARATORS[depth] if depth < len(SEPARATORS) else " "
        segments = _split(chars, sep)

        chunks: list[list[tuple[int, str]]] = []
        current: list[tuple[int, str]] = []

        for seg in segments:
            tentative = current + ([( current[-1][0] if current else seg[0][0], c) for c in sep] if current else []) + seg
            if len(tentative) <= chunk_size:
                current = tentative
            else:
                if current:
                    if len(current) > chunk_size and depth + 1 < len(SEPARATORS):
                        chunks.extend(_recursive_chunk(current, depth + 1))
                    else:
                        chunks.append(current)
                current = seg

        if current:
            if len(current) > chunk_size and depth + 1 < len(SEPARATORS):
                chunks.extend(_recursive_chunk(current, depth + 1))
            else:
                chunks.append(current)

        return chunks

    raw_chunks = _recursive_chunk(stream)

    # Apply overlap
    result: list[tuple[int, str]] = []
    for i, chunk_chars in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev = raw_chunks[i - 1]
            tail = prev[-overlap:]
            chunk_chars = tail + chunk_chars

        page_num   = chunk_chars[0][0]
        chunk_text = "".join(ch for _, ch in chunk_chars).strip()

        if chunk_text:
            result.append((page_num, chunk_text))

    return result






def parse_paper(paper: dict) -> list[dict]:
    """
    Download (if needed) and parse a paper into text chunks
    paper : dict – A paper dict as returned by fetcher.fetch_new_papers()
    Returns list[dict]  –  chunk dicts ready for the indexer
    """
    paper_id = paper["paper_id"]
    pdf_path = _pdf_path(paper_id)


    if not pdf_path.exists():                         # Download only if not already cached
        logger.info("Fetching PDF for %s …", paper_id)
        _download_pdf(paper["pdf_url"], pdf_path)
    else:
        logger.info("PDF already cached for %s skipping download", paper_id)


    logger.info("Extracting text from %s", pdf_path.name)
    pages = _extract_text(pdf_path)
    if not pages:
        logger.warning("No text extracted from %s possibly a scanned PDF", paper_id)
        return []


    logger.info("Chunking %s (%d pages)", paper_id, len(pages))
    chunks = _chunk_text(pages)


    result = [
        {
            "paper_id":  paper_id,
            "title":     paper["title"],
            "chunk_idx": idx,
            "text":      chunk_text,
            "page":      page_num,
        }
        for idx, (page_num, chunk_text) in enumerate(chunks)
    ]

    logger.info(
        "Parsed %s --> %d chunk(s)  (avg %.0f chars)",
        paper_id,
        len(result),
        sum(len(c["text"]) for c in result) / max(len(result), 1),
    )
    return result



# test
if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "%(levelname)s  %(message)s")
    from fetcher import fetch_new_papers

    papers = fetch_new_papers()
    if not papers:
        print("No new papers to test with")

    else:
        test_paper = papers[0]

        print(f"\nParsing: [{test_paper['paper_id']}] {test_paper['title'][:60]}")
        chunks = parse_paper(test_paper)
        print(f"\n✓ {len(chunks)} chunks produced\n")

        for c in chunks[:3]:
            print(f"  chunk {c['chunk_idx']:>3}  page {c['page']}  "
                  f"({len(c['text'])} chars)  {c['text'][:80].strip()!r}")
            
        if len(chunks) > 3:
            print(f"  … {len(chunks) - 3} more chunks")
