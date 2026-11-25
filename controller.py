import time
import uuid
import json
import os
import threading
from datetime import datetime as dt

from main import *   # Jinsy's function

DB_FILE = "data.json"
_LOCK = threading.RLock()

# Default data
USERS = {"user1": {"_id": "user1", "email": "a@b.com", "full_name": "Test User",
                   "fetching_patents": [], "error_patents": []}}
PATENTS = {}


def save():
    with _LOCK:
        with open(DB_FILE, "w") as f:
            json.dump({"users": USERS, "patents": PATENTS}, f, indent=2, default=str)


# Load saved data AFTER USERS / PATENTS exist
if os.path.exists(DB_FILE):
    with open(DB_FILE) as f:
        _d = json.load(f)
    USERS.update(_d.get("users", {}))
    PATENTS.update(_d.get("patents", {}))


def user_exists(user_id: str) -> bool:
    return user_id in USERS


def patent_exists(patent_id: str) -> bool:
    return patent_id in PATENTS


# ---------------- Create user ----------------
def create_user(data: dict) -> dict:
    email = (data.get("email") or "").strip().lower()
    if any(u["email"] == email for u in USERS.values()):
        return {"success": False, "message": "Email already in use"}
    user_id = f"user_{uuid.uuid4().hex[:6]}"
    USERS[user_id] = {
        "_id": user_id,
        "email": email,
        "full_name": data["full_name"],
        "role": data.get("role", "attorney"),
        "created_date": dt.now().strftime("%Y-%m-%d"),
        "fetching_patents": [],
        "error_patents": [],
    }
    save()
    return {"success": True, "message": "User created successfully", "user_id": user_id}


# ---------------- Create patent ----------------
def create_patent(data: dict, user_id: str) -> dict:
    patent_id = str(data.get("_id") or f"local_{uuid.uuid4().hex[:8]}").strip().upper()
    if patent_exists(patent_id):
        return {"success": False, "message": "Patent already exists"}
    PATENTS[patent_id] = {
        **data,
        "_id": patent_id,
        "created_by": user_id,
        "created_date": dt.now().strftime("%Y-%m-%d"),
        "analysis_status": "not_started",
    }
    save()
    return {"success": True, "message": "Patent created successfully", "case_id": patent_id}


# ---------------- Fetch by ID (runs inside the thread) ----------------
def fetch_patent_by_id(app, patent_id: str, user_id: str):
    """Worker that runs inside the thread. Calls Jinsy's function."""
    parent_patent_info_fetching(app, patent_id, user_id)


def start_fetch_thread(app, patent_id: str, user_id: str):
    """Create and start the async thread. Returns (ok, error)."""
    try:
        thread = threading.Thread(
            target=fetch_patent_by_id,
            name=f"{patent_id}-{user_id}",
            args=(app, patent_id, user_id),
            daemon=True,
        )
        thread.start()
        return True, None
    except Exception as e:
        return False, str(e)


# ---------------- Start AI analysis ----------------
def run_ai_analysis(app, case_id: str):
    with app.app_context():
        PATENTS[case_id]["analysis_status"] = "running"
        try:
            time.sleep(1)  # replace with start_patent_analysis / start_product_analysis
            PATENTS[case_id]["analysis_status"] = "completed"
            PATENTS[case_id]["risk_level"] = "low"
        except Exception as e:
            PATENTS[case_id]["analysis_status"] = "error"
            PATENTS[case_id]["error"] = str(e)
        finally:
            save()


def start_analysis_thread(app, case_id: str):
    try:
        t = threading.Thread(target=run_ai_analysis, name=f"analysis-{case_id}",
                             args=(app, case_id), daemon=True)
        t.start()
        return True, None
    except Exception as e:
        return False, str(e)