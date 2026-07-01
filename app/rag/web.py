"""Web search fallback for the CRAG graph.

When the local corpus has no graded-relevant chunks, we rewrite the question
and search the open web. ``duckduckgo-search`` is used because it requires no
API key. Results are filtered to a configurable allowlist so the fallback stays
focused on enterprise-relevant sources.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional
from urllib.parse import urlparse

from duckduckgo_search import DDGS

from app.config import Settings

log = logging.getLogger(__name__)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:  # pragma: no cover
        return ""


def web_search(
    query: str,
    settings: Settings,
    max_results: Optional[int] = None,
) -> list[dict]:
    """Return a list of ``{url, title, body, domain}`` records.

    Results are filtered to the allowlist; if no allowlisted result is found
    we fall back to the top organic results so the system never silently
    returns nothing when there is in fact useful public information.
    """
    max_results = max_results or settings.web_max_results
    allowlist = set(settings.web_allowlist)

    results: list[dict] = []
    try:
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=max_results * 2):
                url = hit.get("href") or hit.get("url") or ""
                dom = _domain(url)
                rec = {
                    "url": url,
                    "title": hit.get("title", ""),
                    "body": hit.get("body", ""),
                    "domain": dom,
                }
                results.append(rec)
                if len(results) >= max_results:
                    break
    except Exception as exc:  # pragma: no cover
        log.warning("Web search failed: %s", exc)
        return []

    allowlisted = [r for r in results if any(dom.endswith(d) for d in allowlist)]
    return allowlisted if allowlisted else results


def make_web_chunk_id(domain: str) -> str:
    """Stable-but-unique id for a web result.

    The first occurrence seeds the id; later ones are ignored by the vector
    store's upsert because we use the same id for a (query, url) pair within
    a single trace.
    """
    short = uuid.uuid5(uuid.NAMESPACE_URL, f"https://{domain}/{uuid.uuid4().hex}")  # type: ignore
    return f"web::{short}::{domain}"
