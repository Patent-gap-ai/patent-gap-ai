"""Database functions for alerts. Used by the MCP tool create_patent and by controller.py.

The alert document has the same fields as _save_alert() in src/mcp_tools/alerts.py.
Right now it is stored in the temporary local store. When MongoDB is ready, rewrite the body of
insert_alert() (e.g. addDataById(connect_to_database(), getAlertDatabaseName(), alert)) and keep its signature.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from memory import mongoDb as store

COLLECTION = "alerts"


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
            "alert_users": list(alert_users),
            "opened_receipts": [],
            "sent_receipts": [],
        }
        try:
            store.insert(COLLECTION, alert)
            return alert["_id"]
        except store.DuplicateKeyError:                       # same millisecond (maybe from another process)
            alert_id += 1


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
