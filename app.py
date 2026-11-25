from flask import Flask, request, jsonify
from flask_cors import CORS

import controller as c

app = Flask(__name__)
CORS(app, origins="*")


def _blank(v) -> bool:
    """True for None / '' / whitespace / 'null' / 'none' / 'undefined'."""
    return v is None or str(v).strip().lower() in ("", "null", "none", "undefined")


def get_user_id():
    """User id from X-User-ID header, else JSON body 'userId'."""
    uid = request.headers.get("X-User-ID")
    if _blank(uid):
        uid = (request.get_json(silent=True) or {}).get("userId")
    return uid


# ---------------- 1. FETCH PATENT BY ID ----------------
@app.route("/api/fetch-patent-by-id", methods=["POST"])
def fetch_patent_by_id():
    data = request.get_json(silent=True) or {}
    user_id = get_user_id()
    patent_id = data.get("patentId")

    if _blank(user_id):
        return jsonify({"success": False, "message": "Invalid User ID"}), 409
    if _blank(patent_id):
        return jsonify({"success": False, "message": "Invalid Patent ID"}), 409
    user_id = str(user_id).strip()
    patent_id = str(patent_id).strip().upper()

    if not c.user_exists(user_id):
        return jsonify({"success": False, "message": "Unauthorized call"}), 401

    if c.patent_exists(patent_id):
        return jsonify({"success": False, "message": "Patent already exists"}), 409

    started, err = c.start_fetch_thread(app, patent_id, user_id)

    if started:
        return jsonify({"success": True,
                        "message": f"Fetching patent {patent_id} has begun"}), 200
    print(f"ERROR: thread start failed: {err}")
    return jsonify({"success": False,
                    "message": "Some internal error occured. Please try again"}), 500


# ---------------- 2. CREATE USER ----------------
@app.route("/api/create-user", methods=["POST"])
def create_user():
    data = request.get_json(silent=True) or {}
    for field, msg in (("email", "Email is required"),
                       ("password", "Password is required"),
                       ("full_name", "Name is required")):
        if _blank(data.get(field)):
            return jsonify({"success": False, "message": msg}), 400
    result = c.create_user(data)
    return jsonify(result), (200 if result["success"] else 400)


# ---------------- 3. CREATE PATENT ----------------
@app.route("/api/create-patent", methods=["POST"])
def create_patent():
    user_id = get_user_id()
    if _blank(user_id):
        return jsonify({"success": False, "message": "User ID is required"}), 400
    if not c.user_exists(str(user_id).strip()):
        return jsonify({"success": False, "message": "Unauthorized call"}), 401
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400
    result = c.create_patent(data, str(user_id).strip())
    return jsonify(result), (200 if result["success"] else 400)


# ---------------- 4. START AI ANALYSIS ----------------
@app.route("/api/start-ai-analysis/<case_id>", methods=["POST"])
def start_ai_analysis(case_id):
    user_id = get_user_id()
    if _blank(user_id):
        return jsonify({"success": False, "message": "Invalid User ID"}), 409
    if not c.user_exists(str(user_id).strip()):
        return jsonify({"success": False, "message": "Unauthorized call"}), 401
    case_id = case_id.strip().upper()
    case = c.PATENTS.get(case_id)
    if case is None:
        return jsonify({"success": False, "message": "Case not found"}), 404
    if case.get("analysis_status") == "running":
        return jsonify({"success": False, "message": "Analysis already running"}), 409

    started, err = c.start_analysis_thread(app, case_id)
    if started:
        return jsonify({"success": True,
                        "message": f"AI analysis for {case_id} has begun"}), 200
    return jsonify({"success": False,
                    "message": "Some internal error occured. Please try again"}), 500


@app.route("/api/case-status/<case_id>")
def case_status(case_id):
    case = c.PATENTS.get(case_id.strip().upper())
    return (jsonify({"success": True, "case": case}), 200) if case else \
           (jsonify({"success": False, "message": "Case not found"}), 404)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
    