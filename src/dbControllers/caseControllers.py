"""Database functions for cases (patents), now backed by MongoDB (src/database.py).

Same four functions, same arguments, same returned values as the temporary-file version,
so controller.py, app.py and the MCP tool (src/mcp_tools/cases.py) need no changes.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from memory.mongoDb import addDataById, connect_to_database, getDataById, updateDataById
from src.envController import getCaseDatabaseName


def _collection() -> str:
    # CASE_DATABASE_NAME_<DEV|PROD|TEST> in .env; falls back to "patents" if it is not set.
    return getCaseDatabaseName() or "patents"


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
        "last_updated": datetime.now(),      # paginateDataByQuery sorts on this field
    }
    added = addDataById(connect_to_database(), _collection(), record)
    # addDataById returns the new id on success, or a "DocumentCreationError..." string on failure.
    if added == case_id:
        return {"success": True, "message": f"Case {case_id} created successfully", "case_id": case_id}
    if "E11000" in str(added) or "duplicate key" in str(added).lower():
        return {"success": False, "message": f"Case {case_id} already exists", "case_id": case_id}
    raise RuntimeError(f"MongoDB insert of case {case_id} failed: {added}")


def get_case(case_id: str) -> dict[str, Any] | None:
    return getDataById(connect_to_database(), _collection(), str(case_id).strip().upper())


def case_exists(case_id: str) -> bool:
    return get_case(case_id) is not None


def update_case(case_id: str, fields: dict[str, Any]) -> bool:
    return updateDataById(connect_to_database(), _collection(), dict(fields),
                          str(case_id).strip().upper())
