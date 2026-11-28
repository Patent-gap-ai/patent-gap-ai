"""Database functions for alerts, now backed by MongoDB (src/database.py).

The alert document has the same fields as _save_alert() in src/mcp_tools/alerts.py.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from memory.mongoDb import addDataById, connect_to_database
from src.envController import getAlertDatabaseName


def _collection() -> str:
    # ALERT_DATABASE_NAME_<DEV|PROD|TEST> in .env; falls back to "alerts" if it is not set.
    return getAlertDatabaseName() or "alerts"


def insert_alert(triggered_by: str, alert_users: list[str], title: str, description: str) -> str:
    """Insert one alert document and return its _id."""
    alert_id = int(time.time() * 1000)                        # milliseconds, like alerts.py
    while True:
        alert: dict[str, Any] = {
            "_id": str(alert_id),
            "title": title,
            "description": description,
            "triggered_by": triggered_by,                     # the case id (= the patent id)
            "triggered_at": datetime.now().strftime("%Y-%m-%d"),
            "last_updated": datetime.now(),
            "alert_users": list(alert_users),
            "opened_receipts": [],
            "sent_receipts": [],
        }
        added = addDataById(connect_to_database(), _collection(), alert)
        if added == alert["_id"]:
            return added
        if "E11000" in str(added) or "duplicate key" in str(added).lower():
            alert_id += 1                                     # same millisecond (maybe another process)
            continue
        raise RuntimeError(f"MongoDB insert of alert failed: {added}")


def insert_case_alert(case_id: str, user_id: str, success: bool, message: str, sources: str = "") -> str:
    """Record the outcome of one create-case attempt (success OR failure) and return the alert _id."""
    if success:
        title = f"Case {case_id} created successfully"
        description = f"Fetched from: {sources or 'unknown'}. Saved to the database."
    else:
        title = f"Case {case_id} could not be created"
        description = str(message)[:500]
    return insert_alert(triggered_by=case_id, alert_users=[user_id] if user_id else [],
                        title=title, description=description)
