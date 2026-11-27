"""MCP server + tool functions.

The FastMCP server object and the @mcp.tool() functions live here.
mcp_tools/mcp.py only imports `mcp` from this file and runs it:  python -m mcp_tools.mcp

Needs these folders from patent-gap-ai copied into the PROJECT ROOT (next to app.py / main.py):
    Sources/USPTO.py
    web_scraper/google_patents.py
    web_scraper/free_patents_online.py
    src/envController.py          (+ your .env file with the USPTO key)

Never print() at module level or in the tool: stdout is the MCP channel. (The tool also
redirects any stray print() from the fetchers to stderr, so they cannot break the protocol.)
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

# This file lives in <project>/mcp_tools/, so the project root is one level up.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.envController import getEnvKey  # noqa: E402
from src.dbControllers.caseControllers import insert_case  # noqa: E402
from src.dbControllers.alertControllers import insert_case_alert  # noqa: E402
from Sources.USPTO import USPTOPatentAPI, USPTOAPIError  # noqa: E402
from web_scraper.google_patents import GooglePatents  # noqa: E402
from web_scraper.free_patents_online import FreePatentsOnline  # noqa: E402

mcp = FastMCP("Patent Case Tools")

# Who owns a new case comes from the environment variable PATENT_GAP_USER_ID, which the Python code that
# starts this server (agent_builder.py) sets. It never comes from the model, so the model cannot pick the user.

# The last fetch of each patent (good or failed), kept in THIS server process.
# create_patent reads it from here, so the model never has to copy a huge JSON by hand.
_LAST_FETCH: dict[str, dict[str, Any]] = {}

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
# Orchestration: fetch from all sources, then validate
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


# --------------------------------------------------------------------------
# The MCP tool
# --------------------------------------------------------------------------
@mcp.tool()
def fetch_by_id(patent_id: str) -> str:
    """Fetch one patent by its id from USPTO, Google Patents and Free Patents Online,
    validate the results, and return everything as a JSON string."""
    try:
        with contextlib.redirect_stdout(sys.stderr):      # stray print() in fetchers must not hit MCP stdout
            result = fetch_patent_details(patent_id)
        _LAST_FETCH[result["patent_id"]] = result         # remembered for create_patent
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 - returned as JSON so the caller can read it
        failed = {"status": "ERROR", "patent_id": patent_id, "error": f"{type(exc).__name__}: {exc}"}
        _LAST_FETCH[(patent_id or "").strip().upper()] = failed       # create_patent reports this reason
        return json.dumps(failed)


@mcp.tool()
def create_patent(patent_id: str) -> str:
    """Insert the patent that fetch_by_id just returned into the database as a new case, then save the
    outcome (success or failure) as an alert. Call it only after fetch_by_id for the same patent id.
    Returns a JSON string with "success" (true/false) and a "message"."""
    key = (patent_id or "").strip().upper()
    label = key or "unknown"
    user_id = os.environ.get("PATENT_GAP_USER_ID", "").strip()
    success, message, sources = False, "", ""

    # 1) insert the case into the database
    try:
        fetched = _LAST_FETCH.get(key)
        if fetched is None:
            message = f"Case {label} was not created: nothing was fetched yet, call fetch_by_id first."
        elif fetched.get("status") == "ERROR":
            message = f"Case {label} was not created: the fetch failed ({fetched.get('error')})."
        elif not fetched.get("validation", {}).get("valid", False):
            message = f"Case {label} was not created: no source returned usable patent details."
        elif not user_id:
            message = f"Case {label} was not created: no user id was given to the MCP server."
        else:
            outcome = insert_case({
                "_id": key,
                "created_by": user_id,
                "fetch_status": "fetched",
                "fetched_data": fetched,
            })
            success, message = outcome["success"], outcome["message"]    # "Case X created successfully"
            sources = ", ".join(fetched["validation"].get("successful_sources", []))
    except Exception as exc:  # noqa: BLE001 - a database error is reported, never crashes the server
        message = f"Case {label} was not created: {type(exc).__name__}: {exc}"

    # 2) insert an alert with the result, whether it worked or not
    alert_id, alert_error = None, None
    try:
        alert_id = insert_case_alert(label, user_id, success, message, sources)
    except Exception as exc:  # noqa: BLE001
        alert_error = f"{type(exc).__name__}: {exc}"

    reply: dict[str, Any] = {"success": success, "message": message, "case_id": key, "alert_id": alert_id}
    if alert_error:
        reply["alert_error"] = alert_error
    return json.dumps(reply)


# ---------------------------------------------------------------------------
# Old hardcoded tools, kept for later. Not registered as MCP tools right now
# (uncomment the @mcp.tool() line above a function to enable it).
# ---------------------------------------------------------------------------
# @mcp.tool()
def get_reference_patent(patent_id: str = "US20240012345A1") -> str:
    """Return the reference patent (the patent being protected) as a JSON string."""
    patent = {
        "patent_id": patent_id,
        "title": "Wireless Power Transfer System",
        "inventors": ["Alex Morgan", "Jamie Lee"],
        "filing_date": "2023-06-15",
        "publication_date": "2024-01-18",
        "abstract": (
            "A system for transferring electrical power wirelessly between "
            "a transmitter and a receiver using resonant coupling."
        ),
        "claims": [
            "A wireless power transfer system comprising: a power transmitter; and a power "
            "receiver configured to receive power from the power transmitter.",
            "The system of claim 1, wherein the receiver uses resonant coupling.",
        ],
    }
    return json.dumps(patent)


# @mcp.tool()
def get_candidate_patent(patent_id: str = "US20240012345A1") -> str:
    """Return the candidate patent/product (the one being compared) as a JSON string.
    `patent_id` is the case id; the candidate has its own patent number."""
    patent = {
        "patent_id": "US20230099876A1",
        "title": "Resonant Wireless Charging Pad",
        "inventors": ["Sam Rivera"],
        "filing_date": "2023-03-02",
        "publication_date": "2023-09-28",
        "abstract": (
            "A charging pad with a transmitter coil that sends power to a receiver coil "
            "in a phone through resonant coupling."
        ),
        "claims": [
            "A charging pad comprising a transmitter coil that sends power wirelessly to a receiver coil.",
            "The charging pad of claim 1, wherein the coils are tuned to the same resonant frequency.",
        ],
    }
    return json.dumps(patent)


# @mcp.tool()
def validate_patent_information(patent_json: str) -> str:
    """Validate that a patent JSON object contains the required fields."""
    required_fields = ["patent_id", "title", "inventors", "filing_date", "abstract", "claims"]

    try:
        patent: Any = json.loads(patent_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"status": "FAILURE", "valid": False, "errors": [f"Invalid JSON: {exc.msg}"]})

    if not isinstance(patent, dict):
        return json.dumps({"status": "FAILURE", "valid": False,
                           "errors": ["Patent information must be a JSON object."]})

    missing_fields = [f for f in required_fields if f not in patent or patent[f] in (None, "", [])]
    errors = [f"Missing or empty required field: {f}" for f in missing_fields]
    return json.dumps({"status": "SUCCESS" if not errors else "FAILURE", "valid": not errors,
                       "checked_fields": required_fields, "errors": errors})