import time
from controller import  PATENTS, USERS, start_fetch_thread          # use your real controller file name

ok, err = start_fetch_thread(None, "US20240012345A1", "user1")
print("thread started:", ok, err)

for i in range(60):                # wait up to 5 minutes
    time.sleep(5)
    if "US20240012345A1" not in USERS["user1"]["fetching_patents"]:
        break
    print("still fetching...")

p = PATENTS.get("US20240012345A1", {})
print("fetch_status:", p.get("fetch_status"))
print("fetch_error :", p.get("fetch_error"))
print("user lists  :", USERS["user1"]["fetching_patents"], USERS["user1"]["error_patents"])