"""Fetch a patent from USPTO, Google Patents and Free Patents Online.

Saved as patent-gap-ai/src/models/cases.py:

    patent-gap-ai/
        Sources/USPTO.py
        web_scraper/google_patents.py
        web_scraper/free_patents_online.py
        src/envController.py
        src/models/cases.py   <- this file

Run it from the project root:
    python src/models/cases.py US1234567
or as a module:
    python -m src.models.cases US1234567

The result is saved automatically to the project root:
    results/<PATENT_ID>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# This file lives in patent-gap-ai/src/models/, so the project root is two
# levels up. Adding it to sys.path lets the imports below work however the
# file is started (terminal, Run button, or python -m).
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.envController import getEnvKey  # noqa: E402
from Sources.USPTO import USPTOPatentAPI, USPTOAPIError  # noqa: E402
from web_scraper.google_patents import GooglePatents  # noqa: E402
from web_scraper.free_patents_online import FreePatentsOnline  # noqa: E402


RESULTS_DIR = ROOT_DIR / "results"
SOURCE_NAMES = ("uspto", "google_patents", "free_patents_online")

SCRAPER_ERRORS = (
    requests.RequestException,
    ValueError,
    IndexError,
    KeyError,
    AttributeError,
    TypeError,
)


# --------------------------------------------------------------------------
# Fetchers: each returns (data, error). Exactly one of them is meaningful.
# --------------------------------------------------------------------------
def _fetch_uspto(patent_id: str) -> tuple[Any, str | None]:
    try:
        api = USPTOPatentAPI(api_key=getEnvKey("uspto"))
        data = api.get_complete_patent_info(patent_id)
    except USPTOAPIError as error:
        return None, str(error)
    if not data:
        return None, "USPTO returned no patent details."
    return data, None


def _fetch_google_patents(patent_id: str) -> tuple[Any, str | None]:
    try:
        data = GooglePatents().search_by_id(patent_id)
    except SCRAPER_ERRORS as error:
        return None, f"{type(error).__name__}: {error}"
    if data is None:
        return None, "Google Patents returned no patent details."
    return data, None


def _fetch_free_patents_online(patent_id: str) -> tuple[Any, str | None]:
    try:
        data = FreePatentsOnline().search_by_id(patent_id)
    except SCRAPER_ERRORS as error:
        return None, f"{type(error).__name__}: {error}"
    if data is None:
        return None, "Free Patents Online returned no patent details."
    return data, None


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _has_source_details(source: str, data: Any) -> bool:
    if source == "uspto" and isinstance(data, dict):
        detail_fields = (
            "title",
            "status",
            "description",
            "filing_date",
            "documents",
            "attorneys",
            "inventors",
            "applicant",
            "current_assignee",
            "other_ids",
        )
        return any(data.get(field) for field in detail_fields)

    if isinstance(data, dict):
        return any(
            value.strip() if isinstance(value, str) else bool(value)
            for value in data.values()
        )
    if isinstance(data, (list, tuple)):
        return any(
            (isinstance(value, str) and value.strip())
            or (isinstance(value, (dict, list, tuple)) and bool(value))
            for value in data
        )
    return isinstance(data, str) and bool(data.strip())


def validate_patent_results(
    patent_id: str, source_results: dict[str, Any], source_errors: dict[str, str]
) -> dict[str, Any]:
    """Report which sources returned usable details."""
    source_status = {
        source: {
            "details_found": _has_source_details(source, source_results.get(source)),
            "error": source_errors.get(source),
        }
        for source in SOURCE_NAMES
    }
    successful_sources = [
        source for source, status in source_status.items() if status["details_found"]
    ]
    return {
        "valid": bool(patent_id.strip()) and bool(successful_sources),
        "successful_sources": successful_sources,
        "source_status": source_status,
        "message": (
            "Patent details were returned by at least one source."
            if successful_sources
            else "No source returned usable patent details."
        ),
    }


# --------------------------------------------------------------------------
# Orchestration and saving
# --------------------------------------------------------------------------
def fetch_patent_details(patent_id: str) -> dict[str, Any]:
    """Fetch a patent from all sources, then validate the results."""
    if not isinstance(patent_id, str) or not patent_id.strip():
        raise ValueError("patent_id must be a non-empty string.")

    normalized_id = patent_id.strip().upper()
    fetchers = {
        "uspto": _fetch_uspto,
        "google_patents": _fetch_google_patents,
        "free_patents_online": _fetch_free_patents_online,
    }
    source_results: dict[str, Any] = {}
    source_errors: dict[str, str] = {}

    for source, fetcher in fetchers.items():
        result, error = fetcher(normalized_id)
        source_results[source] = result
        if error:
            source_errors[source] = error

    return {
        "patent_id": normalized_id,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources": source_results,
        "validation": validate_patent_results(
            normalized_id, source_results, source_errors
        ),
    }


def save_results(result: dict[str, Any]) -> Path:
    """Write the result to results/<PATENT_ID>.json and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", result["patent_id"])
    output_path = RESULTS_DIR / f"{safe_id}.json"
    output_path.write_text(
        json.dumps(result, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch patent details from USPTO, Google Patents, and FPO."
    )
    parser.add_argument("patent_id", help="Patent or application ID to look up")
    args = parser.parse_args()

    try:
        result = fetch_patent_details(args.patent_id)
    except ValueError as error:
        parser.error(str(error))

    output_path = save_results(result)
    validation = result["validation"]
    print(f"{validation['message']}")
    print(f"Successful sources: {validation['successful_sources'] or 'none'}")
    print(f"Saved results to: {output_path}")
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())