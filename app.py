import os
import uuid
import json
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from analyzer import analyze_video

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = "uploads"
app.config["RESULTS_FOLDER"]     = "results"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB hard limit
app.config["SECRET_KEY"]         = os.getenv("SECRET_KEY", "surgiscore-dev-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///surgiscore.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}
DIRECTION_LABELS   = ["north", "south", "east", "west", "internal"]

jobs = {}   # job_id -> {status, result, error, mode}

# Ensure upload/results directories exist at import time
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ── Database & Auth setup ─────────────────────────────────────────────────────

from models_db import db, User, AnalysisResult
db.init_app(app)

from auth import auth_bp, init_oauth, login_required, get_current_user
app.register_blueprint(auth_bp)
init_oauth(app)

with app.app_context():
    db.create_all()


# ── helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Single-camera analysis ────────────────────────────────────────────────────

def run_analysis(job_id, video_path, trainee_name):
    try:
        jobs[job_id]["status"] = "processing"
        result = analyze_video(video_path, app.config["RESULTS_FOLDER"])

        result["previous"] = None
        result["trainee_name"] = trainee_name

        # save report JSON
        result_path = os.path.join(app.config["RESULTS_FOLDER"], f"{job_id}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, default=str)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        print(f"[ERROR] job {job_id}: {e}")




# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    user = None
    if "user" in session:
        user = session["user"]
    return render_template("index.html", user=user)


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file          = request.files["video"]
    trainee_name  = request.form.get("name", "Anonymous").strip() or "Anonymous"

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload MP4, AVI, or MOV only."}), 400

    # check size
    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > 200:
        return jsonify({"error": f"File too large ({size_mb:.0f} MB). Maximum is 200 MB."}), 400

    job_id   = str(uuid.uuid4())[:8]
    filename = secure_filename(f"{job_id}_{file.filename}")
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(video_path)

    jobs[job_id] = {"status": "queued", "result": None, "mode": "single"}
    thread = threading.Thread(
        target=run_analysis,
        args=(job_id, video_path, trainee_name),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found."}), 404
    job  = jobs[job_id]
    resp = {"status": job["status"], "mode": job.get("mode", "single")}
    if job["status"] == "done":
        result = job["result"]
        resp["result"] = {k: v for k, v in result.items()
                          if k not in ("session_data", "rl_summary")}
        if "rl_summary" in result:
            resp["result"]["rl_skill_level"] = result["rl_summary"].get("skill_level")
            resp["result"]["rl_skill_score"] = result["rl_summary"].get("skill_score")
            resp["result"]["rl_target_rate"] = result["rl_summary"].get("target_zone_rate")
    if job["status"] == "error":
        resp["error"] = job.get("error", "Unknown error")
    return jsonify(resp)


@app.route("/results/<job_id>")
def results(job_id):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return "Result not ready.", 404
    result = jobs[job_id]["result"]
    is_logged_in = "user" in session
    return render_template("results.html", result=result, job_id=job_id,
                           is_logged_in=is_logged_in)




@app.route("/save-result/<job_id>", methods=["POST"])
@login_required
def save_result(job_id):
    """Save an analysis result to the logged-in user's account."""
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return jsonify({"error": "Result not found."}), 404

    user = get_current_user()
    if not user:
        return jsonify({"error": "User not found."}), 401

    result = jobs[job_id]["result"]

    # Check if already saved
    existing = AnalysisResult.query.filter_by(user_id=user.id, job_id=job_id).first()
    if existing:
        return jsonify({"message": "Already saved.", "redirect": "/my-dashboard"})

    ar = AnalysisResult(
        user_id          = user.id,
        job_id           = job_id,
        mode             = result.get("mode", "single"),
        stability        = result.get("stability", 0),
        efficiency       = result.get("efficiency", 0),
        precision        = result.get("precision", 0),
        overall          = result.get("overall", 0),
        grade            = result.get("grade", "F"),
        tracking_quality = result.get("tracking_quality", 0),
        frames           = result.get("frames", 0),
        ai_comment       = result.get("ai_comment", ""),
    )
    db.session.add(ar)
    db.session.commit()

    return jsonify({"message": "Result saved!", "redirect": "/my-dashboard"})


@app.route("/my-dashboard")
@login_required
def my_dashboard():
    """Personal performance dashboard."""
    user = get_current_user()
    if not user:
        return redirect(url_for("auth.login"))

    results = AnalysisResult.query.filter_by(user_id=user.id)\
        .order_by(AnalysisResult.created_at.asc()).all()

    # Compute dashboard stats
    if results:
        best_score = max(r.overall for r in results)
        best_grade = max(results, key=lambda r: r.overall).grade
        avg_score  = sum(r.overall for r in results) / len(results)
        latest     = results[-1]
        latest_grade = latest.grade
        latest_date  = latest.created_at.strftime("%Y-%m-%d %H:%M") if latest.created_at else ""
        latest_ai    = latest.ai_comment

        # Skill breakdown (averages)
        def avg_metric(attr):
            vals = [getattr(r, attr) for r in results]
            return sum(vals) / len(vals) if vals else 0

        def score_color(v):
            if v >= 80: return "#00ffb3"
            if v >= 65: return "#00d4ff"
            if v >= 50: return "#ffc820"
            return "#ff3d5a"

        avg_stab = avg_metric("stability")
        avg_eff  = avg_metric("efficiency")
        avg_prec = avg_metric("precision")

        skill_breakdown = [
            ("Stability",  avg_stab, score_color(avg_stab)),
            ("Efficiency", avg_eff,  score_color(avg_eff)),
            ("Precision",  avg_prec, score_color(avg_prec)),
        ]

        # Improvements (first vs latest)
        if len(results) >= 2:
            first = results[0]
            improvements = [
                ("Stability",  latest.stability  - first.stability),
                ("Efficiency", latest.efficiency  - first.efficiency),
                ("Precision",  latest.precision   - first.precision),
                ("Overall",    latest.overall     - first.overall),
            ]
        else:
            improvements = []

        # Chart data
        chart_data = {
            "labels":     [f"Session {i+1}" for i in range(len(results))],
            "overall":    [r.overall for r in results],
            "stability":  [r.stability for r in results],
            "precision":  [r.precision for r in results],
        }
    else:
        best_score = avg_score = 0
        best_grade = latest_grade = "—"
        latest_date = ""
        latest_ai = ""
        skill_breakdown = []
        improvements = []
        chart_data = {"labels": [], "overall": [], "stability": [], "precision": []}

    # Show results in reverse chronological for the table
    results_display = list(reversed(results))

    return render_template("user_dashboard.html",
        user=session["user"],
        results=results_display,
        best_score=best_score,
        best_grade=best_grade,
        avg_score=avg_score,
        latest_grade=latest_grade,
        latest_date=latest_date,
        latest_ai_comment=latest_ai,
        skill_breakdown=skill_breakdown,
        improvements=improvements,
        chart_data=chart_data,
    )


@app.route("/report/<job_id>")
def get_report(job_id):
    return send_from_directory(app.config["RESULTS_FOLDER"], f"{job_id}_report.png")


# ── health check ──────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    """Quick health check endpoint."""
    from analyzer import get_model
    model = get_model()
    return jsonify({
        "status": "ok",
        "yolo_model": "YOLOv8" if model != "fallback" else "fallback",
        "active_jobs": len(jobs),
        "jobs_by_status": {
            s: sum(1 for j in jobs.values() if j["status"] == s)
            for s in ["queued", "processing", "done", "error"]
        },
    })


# ── error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum allowed size is 200 MB."}), 413


# ── startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
