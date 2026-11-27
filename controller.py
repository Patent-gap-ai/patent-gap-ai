import copy
import time
import uuid
import threading
from datetime import datetime as dt

from main import parent_patent_info_fetching   # Jinsy's function
from memory import mongoDb as store
from src.dbControllers.caseControllers import case_exists, get_case, insert_case, update_case
from src.dbControllers.alertControllers import insert_case_alert

_LOCK = threading.RLock()

# Users still live in memory here (saved to the "users" part of data.json).
# Patents and alerts are NOT kept in memory: the MCP tool inserts them from its own process, so Flask
# always reads them from the database layer (src/dbControllers).
USERS = {"user1": {"_id": "user1", "email": "a@b.com", "full_name": "Test User",
                   "fetching_patents": [], "error_patents": []}}


def save():
    """Save the users. (Only the users: patents and alerts are written by the database layer.)"""
    with _LOCK:
        store.replace_section("users", copy.deepcopy(USERS))


# Load saved users AFTER USERS exists
USERS.update(store.read_all().get("users", {}))


def user_exists(user_id: str) -> bool:
    return user_id in USERS


def patent_exists(patent_id: str) -> bool:
    return case_exists(patent_id)


def get_patent(patent_id: str):
    return get_case(patent_id)


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


# ---------------- Create patent (manual, from the /api/create-patent route) ----------------
def create_patent(data: dict, user_id: str) -> dict:
    patent_id = str(data.get("_id") or f"local_{uuid.uuid4().hex[:8]}").strip().upper()
    result = insert_case({**data, "_id": patent_id, "created_by": user_id})
    if not result["success"]:
        return {"success": False, "message": "Patent already exists"}
    return {"success": True, "message": "Patent created successfully", "case_id": patent_id}


# ---------------- Fetch by ID (runs inside the thread) ----------------
def console_status(patent_id: str, error, message: str = "") -> None:
    """Print one clear line so the person watching the server console sees the outcome."""
    when = dt.now().strftime("%H:%M:%S")
    if error is None:
        print(f"[{when}] [create-patent] {patent_id} -> CREATED: {message or 'Case created successfully'}",
              flush=True)
    else:
        reason = " ".join(str(error).split())[:300]            # one line, not too long
        print(f"[{when}] [create-patent] {patent_id} -> NOT CREATED: {reason}", flush=True)


def fetch_patent_by_id(app, patent_id: str, user_id: str):
    """Worker that runs inside the thread.

    The AI run fetches the patent, and the create_patent MCP tool inserts the case into the database and
    inserts an alert with the result. This function only tracks progress for the user. The thread is
    released when this function returns, and the `finally` block guarantees that happens with the
    "fetching" flag cleared, whatever went wrong.
    """
    patent_id = patent_id.strip().upper()

    # mark as fetching
    with _LOCK:
        user = USERS.get(user_id)
        if user and patent_id not in user["fetching_patents"]:
            user["fetching_patents"].append(patent_id)
        save()

    db, error = {}, None
    try:
        result = parent_patent_info_fetching(app, patent_id, user_id)
        db = result.get("db_insert") or {}
        if not db.get("success"):
            error = db.get("message") or "The case was not created"
    except Exception as e:  # noqa: BLE001 - recorded below, never lost
        error = f"{type(e).__name__}: {e}"
    finally:
        console_status(patent_id, error, db.get("message", ""))

        # The tool normally records the alert. If it could not (the run crashed before the tool was called,
        # or the tool's alert insert failed), record it here so no attempt goes without an alert.
        if not db.get("alert_id"):
            try:
                insert_case_alert(patent_id, user_id, error is None,
                                  error or db.get("message", ""), "")
            except Exception as e:  # noqa: BLE001
                print(f"ERROR: could not save the alert for {patent_id}: {e}", flush=True)

        if error is not None:
            # keep a record of the failure, but never replace a case that already exists
            try:
                insert_case({"_id": patent_id, "created_by": user_id,
                             "fetch_status": "error", "fetch_error": error})
            except Exception as e:  # noqa: BLE001
                print(f"ERROR: could not record the failure for {patent_id}: {e}", flush=True)

        with _LOCK:
            user = USERS.get(user_id)
            if user and error is not None and patent_id not in user["error_patents"]:
                user["error_patents"].append(patent_id)
            if user and patent_id in user["fetching_patents"]:
                user["fetching_patents"].remove(patent_id)
            save()
        # leaving this function ends the thread: it is released


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
        update_case(case_id, {"analysis_status": "running"})
        try:
            time.sleep(1)  # replace with start_patent_analysis / start_product_analysis
            update_case(case_id, {"analysis_status": "completed", "risk_level": "low"})
        except Exception as e:
            update_case(case_id, {"analysis_status": "error", "error": str(e)})


def start_analysis_thread(app, case_id: str):
    try:
        t = threading.Thread(target=run_ai_analysis, name=f"analysis-{case_id}",
                             args=(app, case_id), daemon=True)
        t.start()
        return True, None
    except Exception as e:
        return False, str(e)
