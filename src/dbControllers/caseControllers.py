"""Database functions for cases (patents). Used by the MCP tool create_patent and by controller.py.

Right now they use the temporary local store. When MongoDB is ready, rewrite the bodies of these four
functions (keep the names, arguments and returned values) and nothing else in the project has to change.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from memory import mongoDb as store

COLLECTION = "patents"


def insert_case(case: dict[str, Any]) -> dict[str, Any]:
    """Insert one new case. Returns {"success": bool, "message": str, "case_id": str}.

    A duplicate id is a normal failure (success False). Real database errors are raised."""
    case_id = str(case.get("_id") or "").strip().upper()
    if not case_id:
        return {"success": False, "message": "Case id is required", "case_id": ""}
    record = {
        "analysis_status": "not_started",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        **case,
        "_id": case_id,
    }
    try:
        store.insert(COLLECTION, record)
    except store.DuplicateKeyError:
        return {"success": False, "message": f"Case {case_id} already exists", "case_id": case_id}
    return {"success": True, "message": f"Case {case_id} created successfully", "case_id": case_id}


def get_case(case_id: str) -> dict[str, Any] | None:
    return store.get(COLLECTION, str(case_id).strip().upper())


def case_exists(case_id: str) -> bool:
    return get_case(case_id) is not None


def update_case(case_id: str, fields: dict[str, Any]) -> bool:
    return store.update_fields(COLLECTION, str(case_id).strip().upper(), fields)