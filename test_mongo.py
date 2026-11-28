"""Run:  python test_mongo.py     (from the project root, with .env filled in)

Checks the connection, then writes, reads, updates and deletes one throw-away case and alert
in the REAL database, so you can see that values are stored."""
from memory.mongoDb import connect_to_database, deleteDataById
from src.envController import getCaseDatabaseName, getAlertDatabaseName
from src.dbControllers.caseControllers import insert_case, get_case, update_case
from src.dbControllers.alertControllers import insert_case_alert

db = connect_to_database()
print("Connected to database:", db.name)
print("Case collection :", getCaseDatabaseName() or "patents")
print("Alert collection:", getAlertDatabaseName() or "alerts")

print("insert :", insert_case({"_id": "TEST-0001", "title": "Smoke test"}))
print("again  :", insert_case({"_id": "TEST-0001"}), "(duplicate should fail)")
print("update :", update_case("TEST-0001", {"analysis_status": "completed"}))
print("get    :", get_case("TEST-0001"))
alert_id = insert_case_alert("TEST-0001", "user1", True, "", "test")
print("alert  :", alert_id)

input("\nOpen Compass / Atlas now and look for TEST-0001. Press Enter to delete the test data...")
deleteDataById(db, getCaseDatabaseName() or "patents", "TEST-0001")
deleteDataById(db, getAlertDatabaseName() or "alerts", alert_id)
print("cleaned up")
