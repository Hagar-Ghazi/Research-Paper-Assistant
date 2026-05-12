"""
Fetches new arXiv papers for every topic in config.ARXIV_TOPICS
fetch_new_papers() -> list[dict]
    Returns a list of paper dicts
    Papers already recorded in INGESTED_FILE are silently skipped

Paper dict keys
---------------
    paper_id  : str   
    title     : str
    authors   : list[str]
    date      : str       (published date)
    abstract  : str
    pdf_url   : str       (direct link to the PDF)
"""

import re
import time
import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
import config
logger = logging.getLogger(__name__)


# arXiv API constants
_BASE_URL   = "http://export.arxiv.org/api/query"
_NS         = {"atom": "http://www.w3.org/2005/Atom"}
_RETRY_MAX  = 4
_RETRY_WAIT = 20


def _load_ingested() -> set[str]:
    """Return the set of paper_ids already recorded in INGESTED_FILE"""
    path = config.INGESTED_FILE
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}




def _extract_id(entry_id: str) -> str:
    """
    arXiv entry <id> looks like:
        http://arxiv.org/abs/2401.12345v2
    We want just '2401.12345' (no version suffix)
    """
    match = re.search(r"abs/([^v]+)", entry_id)
    return match.group(1) if match else entry_id.split("/")[-1]




def _parse_entry(entry: ET.Element) -> dict:
    """Convert one Atom <entry> element into a paper dict"""
    def text(tag: str) -> str:
        el = entry.find(f"atom:{tag}", _NS)
        return el.text.strip() if el is not None and el.text else ""

    raw_id  = text("id")
    paper_id = _extract_id(raw_id)

    authors = [
        (a.find("atom:name", _NS).text or "").strip()
        for a in entry.findall("atom:author", _NS)
        if a.find("atom:name", _NS) is not None
    ]

    
    published = text("published")
    date = published[:10] if published else ""
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"


    return {
        "paper_id": paper_id,
        "title":    " ".join(text("title").split()),   
        "authors":  authors,
        "date":     date,
        "abstract": " ".join(text("summary").split()),
        "pdf_url":  pdf_url,
    }




def _fetch_topic(topic: str, max_results: int, ingested: set[str]) -> list[dict]:
    """
    Query arXiv for `topic`return up to `max_results` new paper dicts
    (skipping IDs already in `ingested`)
    """
    params = urllib.parse.urlencode({
        "search_query": f"cat:{topic}",
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
        "max_results":  max_results * 3,
        "start":        0,
    })

    url = f"{_BASE_URL}?{params}"
    xml_bytes = None


    for attempt in range(1, _RETRY_MAX + 1):
        try:
            with urllib.request.urlopen(url, timeout = 60) as resp:
                xml_bytes = resp.read()
            break
        except Exception as exc:
            logger.warning("arXiv request failed (attempt %d/%d): %s", attempt, _RETRY_MAX, exc)
            if attempt < _RETRY_MAX:
                wait = _RETRY_WAIT * attempt
                logger.warning("Waiting %ds before retry …", wait)
                time.sleep(wait)
            else:
                logger.error("Giving up on topic %s after %d attempts.", topic, _RETRY_MAX)
                return []

    time.sleep(20)



    root    = ET.fromstring(xml_bytes)
    entries = root.findall("atom:entry", _NS)
    new_papers: list[dict] = []

    for entry in entries:
        paper = _parse_entry(entry)
        if paper["paper_id"] in ingested:
            logger.debug("Skipping already-ingested paper: %s", paper["paper_id"])
            continue
        new_papers.append(paper)
        if len(new_papers) >= max_results:
            break

    logger.info("Topic %-12s --> %d new paper(s) found", topic, len(new_papers))
    return new_papers





def fetch_new_papers() -> list[dict]:
    """
    Iterate over config.ARXIV_TOPICS, query arXiv and return all new papers
    as a flat list of dicts
    Papers already in INGESTED_FILE are skipped
    Duplicates across topics (same paper_id) are deduplicated
    """
    ingested = _load_ingested()
    logger.info("Loaded %d already-ingested paper IDs", len(ingested))

    all_papers: list[dict] = []
    seen_this_run: set[str] = set()

    for topic in config.ARXIV_TOPICS:
        papers = _fetch_topic(topic, config.MAX_PAPERS_PER_TOPIC, ingested | seen_this_run)
        for p in papers:
            seen_this_run.add(p["paper_id"])
            all_papers.append(p)

    logger.info("Total new papers fetched across all topics: %d", len(all_papers))
    return all_papers



if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "%(levelname)s  %(message)s")
    papers = fetch_new_papers()
    for p in papers:
        print(f"[{p['date']}] {p['paper_id']:>15}  {p['title'][:70]}")
