"""TEMPORARY local-file database, shared by the Flask app and the MCP server process.

Why it exists: the real MongoDB layer (src/database.py) is not written yet. The MCP tool runs in its own
process, so it cannot share Python dictionaries with Flask. Both processes therefore read and write the
same file (data.json), one small locked transaction at a time:  lock -> read -> change -> write -> unlock.
Nobody keeps their own copy of patents or alerts in memory, so nobody can overwrite anybody else's insert.

Collections are top-level keys of data.json:  "users", "patents", "alerts".
When MongoDB is ready, only the functions in src/dbControllers/*.py need to change. This file can go.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = Path(os.environ.get("PATENT_GAP_DATA_FILE") or ROOT_DIR / "data.json")
_LOCK = FileLock(str(DATA_FILE) + ".lock", timeout=30)      # works across processes (and Windows/macOS/Linux)


class DuplicateKeyError(Exception):
    """A document with this _id already exists in the collection."""


def _read() -> dict[str, Any]:
    if not DATA_FILE.exists():
        return {}
    text = DATA_FILE.read_text(encoding="utf-8")
    return json.loads(text) if text.strip() else {}


def _write(data: dict[str, Any]) -> None:
    tmp = DATA_FILE.with_name(DATA_FILE.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, DATA_FILE)                               # atomic: readers never see a half-written file


def transaction(change: Callable[[dict[str, Any]], Any]) -> Any:
    """Run change(data) while holding the lock, then save. If change() raises, nothing is saved."""
    with _LOCK:
        data = _read()
        result = change(data)
        _write(data)
        return result


def read_all() -> dict[str, Any]:
    with _LOCK:
        return _read()


def insert(collection: str, doc: dict[str, Any]) -> None:
    """Add one document. Raises DuplicateKeyError if its _id is already taken."""
    def _do(data: dict[str, Any]) -> None:
        section = data.setdefault(collection, {})
        if doc["_id"] in section:
            raise DuplicateKeyError(f"{collection}/{doc['_id']} already exists")
        section[doc["_id"]] = doc
    transaction(_do)


def get(collection: str, doc_id: str) -> dict[str, Any] | None:
    return read_all().get(collection, {}).get(doc_id)


def update_fields(collection: str, doc_id: str, fields: dict[str, Any]) -> bool:
    """Change some fields of an existing document. Returns False if it does not exist."""
    def _do(data: dict[str, Any]) -> bool:
        doc = data.get(collection, {}).get(doc_id)
        if doc is None:
            return False
        doc.update(fields)
        return True
    return transaction(_do)


def replace_section(collection: str, content: dict[str, Any]) -> None:
    transaction(lambda data: data.__setitem__(collection, content))
