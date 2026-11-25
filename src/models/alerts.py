#Neethu
"""Look up a patent ID and save an alert to the database either way.

Location: patent-gap-ai/src/models/alerts.py

Outcomes (each one saves an alert):
    found         - at least one source returned details for the ID
    not_found     - sources answered, but none had the patent
    lookup_failed - every source raised an error (network, key, parsing), so we
                    cannot say whether the patent exists

Usage (from the project root):
    python -c "from src.models.alerts import alert_for_patent_lookup; print(alert_for_patent_lookup('US1234567', user_id='u123'))"
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# This file lives in patent-gap-ai/src/models/, so the project root is two
# levels up. Adding it to sys.path makes the imports below work however the
# file is started.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Adjust these imports if your folders or function names differ.
from src.database import connect_to_database, addDataById  # noqa: E402
from src.envController import getAlertDatabaseName, getEnvKey  # noqa: E402
from Sources.USPTO import USPTOPatentAPI, USPTOAPIError  # noqa: E402
from Web_Scraper.google_patents import GooglePatents  # noqa: E402
from Web_Scraper.free_patents_online import FreePatentsOnline  # noqa: E402


SCRAPER_ERRORS = (
    requests.RequestException,
    ValueError,
    IndexError,
    KeyError,
    AttributeError,
    TypeError,
)


# --------------------------------------------------------------------------
# Searching the sources
# --------------------------------------------------------------------------
def _has_data(data: Any) -> bool:
    """True if a source returned something non-empty."""
    if data is None:
        return False
    if isinstance(data, str):
        return bool(data.strip())
    if isinstance(data, dict):
        return any(
            v.strip() if isinstance(v, str) else bool(v) for v in data.values()
        )
    if isinstance(data, (list, tuple)):
        return len(data) > 0
    return bool(data)


def _query_uspto(patent_id: str) -> Any:
    api = USPTOPatentAPI(api_key=getEnvKey("uspto"))
    return api.get_complete_patent_info(patent_id)


def _query_google(patent_id: str) -> Any:
    return GooglePatents().search_by_id(patent_id)


def _query_fpo(patent_id: str) -> Any:
    return FreePatentsOnline().search_by_id(patent_id)


SOURCES = {
    "uspto": _query_uspto,
    "google_patents": _query_google,
    "free_patents_online": _query_fpo,
}


def search_patent_sources(patent_id: str) -> dict[str, Any]:
    """Ask every source for the patent and report where it was found."""
    found_in: list[str] = []
    errors: dict[str, str] = {}
    details: dict[str, Any] = {}

    for name, query in SOURCES.items():
        try:
            data = query(patent_id)
        except (USPTOAPIError, *SCRAPER_ERRORS) as error:
            errors[name] = f"{type(error).__name__}: {error}"
            continue
        if _has_data(data):
            found_in.append(name)
            details[name] = data

    if found_in:
        status = "found"
    elif len(errors) == len(SOURCES):
        status = "lookup_failed"
    else:
        status = "not_found"

    return {"status": status, "found_in": found_in, "errors": errors, "details": details}


def _extract_title(details: dict[str, Any]) -> str:
    """Best-effort patent title from whichever source has one."""
    for data in details.values():
        if isinstance(data, dict):
            title = data.get("title")
            if isinstance(title, str) and title.strip():
                return title.strip()
    return ""


# --------------------------------------------------------------------------
# Saving the alert
# --------------------------------------------------------------------------
def _save_alert(
    triggered_by: str,
    triggered_at: str,
    alert_users: list[str],
    title: str,
    description: str,
) -> str:
    """Write one alert document to the alerts collection and return its _id."""
    new_alert = {
        # Milliseconds, so two alerts in the same second do not collide.
        "_id": str(int(time.time() * 1000)),
        "title": title,
        "description": description,
        "triggered_by": triggered_by,
        "triggered_at": triggered_at,
        "alert_users": alert_users,
        "opened_receipts": [],
        "sent_receipts": [],
    }
    addDataById(connect_to_database(), getAlertDatabaseName(), new_alert)
    return new_alert["_id"]


def alert_for_patent_lookup(
    patent_id: str,
    user_id: str,
    case_id: str | None = None,
    extra_alert_users: list[str] | None = None,
) -> dict[str, Any]:
    """Search for the patent and save an alert. Returns status, alert_id and sources.

    case_id: the case this lookup belongs to, stored in 'triggered_by'.
             If omitted, user_id is stored instead.
    """
    patent_id = (patent_id or "").strip().upper()
    if not patent_id or not user_id:
        raise ValueError("patent_id and user_id are required.")

    result = search_patent_sources(patent_id)
    status = result["status"]

    if status == "found":
        sources = ", ".join(result["found_in"])
        title_text = _extract_title(result["details"])
        title = f"Patent {patent_id} found"
        description = f"Found in: {sources}."
        if title_text:
            description += f" Title: {title_text}."
    elif status == "not_found":
        title = f"Patent {patent_id} not found"
        description = "No source returned details for this patent ID."
    else:
        title = f"Patent {patent_id} lookup failed"
        failed = "; ".join(f"{s}: {e}" for s, e in result["errors"].items())
        description = f"All sources failed, so the result is unknown. {failed}"

    alert_users = [user_id] + [u for u in (extra_alert_users or []) if u != user_id]

    alert_id = _save_alert(
        triggered_by=case_id or user_id,
        triggered_at=datetime.now().strftime("%Y-%m-%d"),
        alert_users=alert_users,
        title=title,
        description=description,
    )

    return {
        "patent_id": patent_id,
        "status": status,
        "found_in": result["found_in"],
        "errors": result["errors"],
        "alert_id": alert_id,
    }