from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date
from typing import Any

import httpx

from app.config import get_settings
from app.models import PaperCandidate

ATOM = "http://www.w3.org/2005/Atom"
NS = {"atom": ATOM}


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def parse_arxiv_feed(xml_payload: str) -> list[PaperCandidate]:
    """Parse an arXiv Atom response into validated paper candidates."""

    root = ET.fromstring(xml_payload)
    papers: list[PaperCandidate] = []
    for entry in root.findall("atom:entry", NS):
        entry_id = _clean_text(entry.findtext("atom:id", default="", namespaces=NS))
        arxiv_id = entry_id.rsplit("/abs/", 1)[-1]
        if not arxiv_id:
            continue

        published_raw = _clean_text(
            entry.findtext("atom:published", default="", namespaces=NS)
        )
        published_at = date.fromisoformat(published_raw[:10])
        updated_raw = _clean_text(
            entry.findtext("atom:updated", default="", namespaces=NS)
        )
        links = entry.findall("atom:link", NS)
        pdf_url = next(
            (
                link.attrib.get("href")
                for link in links
                if link.attrib.get("title") == "pdf"
            ),
            None,
        )
        authors = [
            _clean_text(author.findtext("atom:name", default="", namespaces=NS))
            for author in entry.findall("atom:author", NS)
        ]
        papers.append(
            PaperCandidate(
                arxiv_id=arxiv_id,
                title=_clean_text(
                    entry.findtext("atom:title", default="", namespaces=NS)
                ),
                abstract=_clean_text(
                    entry.findtext("atom:summary", default="", namespaces=NS)
                ),
                authors=[author for author in authors if author],
                arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_url,
                published_at=published_at,
                updated_at=date.fromisoformat(updated_raw[:10])
                if updated_raw
                else None,
            )
        )
    return papers


async def search_arxiv(
    query: str,
    start_date: str | None = None,
    end_date: str | None = None,
    max_results: int = 30,
) -> list[dict[str, Any]]:
    """Search arXiv and return normalized paper metadata.

    This is an ADK-compatible tool. Date filtering is repeated locally because
    source APIs can return records with incomplete or inconsistent date data.
    """

    settings = get_settings()
    params = {
        "search_query": f'all:"{query.strip()}"',
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(settings.arxiv_api_url, params=params)
        response.raise_for_status()

    papers = parse_arxiv_feed(response.text)
    if start_date:
        lower_bound = date.fromisoformat(start_date)
        papers = [paper for paper in papers if paper.published_at >= lower_bound]
    if end_date:
        upper_bound = date.fromisoformat(end_date)
        papers = [paper for paper in papers if paper.published_at <= upper_bound]
    return [paper.model_dump(mode="json") for paper in papers]
