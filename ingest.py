"""
Orchestrates the full ingestion pipeline:
    fetch  ->  parse  ->  index  ->  record

run() -> None
    Fetch new papers from arXiv parse each into chunks
    add them to the indexes and record their IDs in INGESTED_FILE
    so they are skipped on the next run
"""

import logging
import config
from fetcher  import fetch_new_papers
from parser   import parse_paper
from indexer  import add_chunks
logger = logging.getLogger(__name__)



def _record_ingested(paper_ids: list[str]) -> None:
    config.INGESTED_FILE.parent.mkdir(parents = True, exist_ok = True)
    with config.INGESTED_FILE.open("a") as f:
        for pid in paper_ids:
            f.write(pid + "\n")



def run() -> None:
    config.CACHE_DIR.mkdir(parents = True, exist_ok = True)
    config.PDF_DIR.mkdir(parents = True, exist_ok = True)

    logger.info("Fetching new papers")
    papers = fetch_new_papers()


    if not papers:
        logger.info("No new papers found nothing to ingest")
        return

    logger.info("%d new paper(s) to ingest", len(papers))



    ingested_ids: list[str] = []
    failed_ids:   list[str] = []

    for paper in papers:
        pid = paper["paper_id"]
        logger.info("Processing [%s] %s", pid, paper["title"][:60])

        try:
            chunks = parse_paper(paper)
            if not chunks:
                logger.warning("No chunks produced for %s skipping", pid)
                failed_ids.append(pid)
                continue

            add_chunks(chunks)
            ingested_ids.append(pid)
            logger.info("Indexed %d chunks for %s", len(chunks), pid)

        except Exception as exc:
            logger.error("Failed to ingest %s: %s", pid, exc, exc_info=True)
            failed_ids.append(pid)

            

    if ingested_ids:
        _record_ingested(ingested_ids)
        logger.info("Recorded %d ingested paper ID(s)", len(ingested_ids))

    logger.info(
        "Ingestion complete  success: %d  failed: %d",
        len(ingested_ids),
        len(failed_ids),
    )

    if failed_ids:
        logger.warning("Failed paper IDs: %s", failed_ids)


if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
    )
    run()
