"""Run:  python test_controller.py [PATENT_ID]

Fetches one patent through the background thread, the model and the MCP tools. The create_patent tool
inserts the case and an alert straight into the database, so the Flask server does NOT need to run.
"""
import sys
import threading
import time

from controller import USERS, get_patent, start_fetch_thread
from memory.mongoDb import connect_to_database, getDataByKeyValue
from src.envController import getAlertDatabaseName

PID = (sys.argv[1] if len(sys.argv) > 1 else "US20240012345A1").strip().upper()

ok, err = start_fetch_thread(None, PID, "user1")
print("thread started:", ok, err)

thread = next(t for t in threading.enumerate() if t.name == f"{PID}-user1")
start = time.time()
while thread.is_alive() and time.time() - start < 600:       # wait up to 10 minutes
    time.sleep(5)
    print("still working...")

p = get_patent(PID) or {}
print("\nthread alive    :", thread.is_alive(), "(False = released)")
print("case in database:", bool(p))
print("fetch_status    :", p.get("fetch_status"))
print("fetch_error     :", p.get("fetch_error"))
print("has fetched_data:", "fetched_data" in p)
print("fetching list   :", USERS["user1"]["fetching_patents"])
print("error list      :", USERS["user1"]["error_patents"])
print("\nalerts for this case:")
for a in getDataByKeyValue(connect_to_database(), getAlertDatabaseName() or "alerts", "triggered_by", PID):
    print("  -", a["title"], "|", a["description"][:120])