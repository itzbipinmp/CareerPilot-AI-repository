"""CareerPilot AI: a small, local-first Flask career assistant."""

from __future__ import annotations

import os
import re
import secrets
import sqlite3
import hmac
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_file, session, url_for
from werkzeug.utils import secure_filename

from careerpilot import FIT_SCORE_VERSION, analyze_career_fit, build_resume_feedback, extract_resume_text, get_llm_settings, match_job, retrieve_evidence
from resume_export import make_resume_pdf
from storage import delete_analysis, get_analysis, init_db, list_analyses, save_analysis
from storage import update_analysis_report

BASE_DIR = Path(__file__).resolve().parent


def _load_env_file(path: Path) -> None:
    """Read simple KEY=value settings without exposing them to the browser."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


# Load private, server-side settings before Flask reads its secret and AI config.
_load_env_file(BASE_DIR / ".env")
DB_PATH = Path(os.environ.get("CAREERPILOT_DB", BASE_DIR / "careerpilot.db"))
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_RESUME_BYTES = 8 * 1024 * 1024

app = Flask(__name__)
secret_key = os.environ.get("FLASK_SECRET_KEY")
if os.environ.get("APP_ENV") == "production" and not secret_key:
    raise RuntimeError("Set FLASK_SECRET_KEY before starting CareerPilot AI in production.")
app.config["SECRET_KEY"] = secret_key or "local-development-key-change-me"
app.config["MAX_CONTENT_LENGTH"] = 9 * 1024 * 1024  # Leaves room for the multipart form around an 8 MiB file.
app.config["DATABASE"] = str(DB_PATH)
app.config["LLM_SETTINGS"] = get_llm_settings()
init_db(DB_PATH)


@app.before_request
def protect_form_posts():
    if request.method == "POST":
        stored_token = session.get("csrf_token", "")
        submitted_token = request.form.get("csrf_token", "")
        if not stored_token or not submitted_token or not hmac.compare_digest(stored_token, submitted_token):
            abort(400)
    else:
        session.setdefault("csrf_token", secrets.token_urlsafe(32))


@app.context_processor
def inject_csrf_token():
    return {"csrf_token": session.get("csrf_token", "")}


def _ensure_fit_assessment(analysis: dict) -> None:
    """Upgrade old saved keyword scores to the evidence-weighted rubric once."""
    report = analysis["report"]
    if report.get("score_version") == FIT_SCORE_VERSION and report.get("score_breakdown"):
        return
    refreshed = match_job(analysis["resume_text"], analysis["job_text"], analysis["job_title"])
    report.update({
        key: refreshed[key]
        for key in ("match_score", "summary", "matched_skills", "evidence", "score_breakdown", "score_method", "score_version")
    })
    report["skill_gaps"] = [
        {"skill": skill, "next_step": f"Add a real project or course demonstrating {skill}, then describe your contribution and result."}
        for skill in refreshed["missing_skills"][:8]
    ]
    update_analysis_report(DB_PATH, analysis["id"], report)


def _recent_analyses(limit: int) -> list[dict]:
    analyses = list_analyses(DB_PATH, limit=limit)
    for item in analyses:
        full = get_analysis(DB_PATH, item["id"])
        if full:
            _ensure_fit_assessment(full)
    return list_analyses(DB_PATH, limit=limit)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def render_workspace(job_title: str = "", job_description: str = "", status: int = 200):
    return render_template(
        "index.html",
        analyses=_recent_analyses(limit=5),
        llm_enabled=app.config["LLM_SETTINGS"]["enabled"],
        llm_provider={"openai": "OpenAI", "gemini": "Gemini"}.get(str(app.config["LLM_SETTINGS"]["provider"]), "Local tools"),
        job_title=job_title,
        job_description=job_description,
    ), status


@app.get("/")
def index():
    return render_workspace()


@app.get("/health")
def health():
    return {"status": "ok"}, 200


@app.post("/analyze")
def analyze():
    uploaded = request.files.get("resume")
    job_text = request.form.get("job_description", "").strip()
    job_title = request.form.get("job_title", "").strip() or "Untitled role"

    if not uploaded or not uploaded.filename:
        flash("Choose a resume file to continue.", "error")
        return render_workspace(job_title, job_text, status=400)
    if not allowed_file(uploaded.filename):
        flash("Please upload a PDF, DOCX, or TXT resume.", "error")
        return render_workspace(job_title, job_text, status=400)
    if len(job_text) < 80:
        flash("Add a job description with at least 80 characters for a useful analysis.", "error")
        return render_workspace(job_title, job_text, status=400)

    try:
        resume_bytes = uploaded.read()
        if len(resume_bytes) > MAX_RESUME_BYTES:
            raise ValueError("That file is larger than 8 MB. Try a smaller resume.")
        resume_text = extract_resume_text(resume_bytes, secure_filename(uploaded.filename))
        if len(resume_text.strip()) < 40:
            raise ValueError("I couldn't find enough readable text in that file. Try a text-based PDF, DOCX, or TXT file.")
        report = analyze_career_fit(resume_text, job_text, job_title, app.config["LLM_SETTINGS"])
    except (ValueError, sqlite3.Error) as exc:
        flash(str(exc), "error")
        return render_workspace(job_title, job_text, status=400)
    except Exception:
        app.logger.exception("Career brief generation failed")
        flash("I couldn’t finish that analysis. Check the resume file and try again.", "error")
        return render_workspace(job_title, job_text, status=500)

    analysis_id = save_analysis(
        DB_PATH,
        job_title=job_title,
        resume_name=secure_filename(uploaded.filename) or "resume",
        resume_text=resume_text,
        job_text=job_text,
        report=report,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return redirect(url_for("result", analysis_id=analysis_id))


@app.get("/analysis/<int:analysis_id>")
def result(analysis_id: int):
    analysis = get_analysis(DB_PATH, analysis_id)
    if analysis is None:
        abort(404)
    _ensure_fit_assessment(analysis)
    _ensure_resume_draft(analysis)
    # Rebuild older saved snippets with cleaned, short passages so noisy PDF list
    # glyphs and unrelated sections do not flood the evidence card.
    analysis["report"]["evidence"] = retrieve_evidence(analysis["resume_text"], analysis["job_text"])
    provider = str(app.config["LLM_SETTINGS"]["provider"])
    return render_template(
        "result.html",
        analysis=analysis,
        llm_enabled=app.config["LLM_SETTINGS"]["enabled"],
        llm_provider={"openai": "OpenAI", "gemini": "Gemini"}.get(provider, "AI"),
    )


@app.post("/analysis/<int:analysis_id>/retry-ai")
def retry_ai_analysis(analysis_id: int):
    """Retry an AI review using the saved documents after a provider failure."""
    analysis = get_analysis(DB_PATH, analysis_id)
    if analysis is None:
        abort(404)
    settings = app.config["LLM_SETTINGS"]
    if not settings.get("enabled"):
        flash("No AI provider is configured on this server yet.", "error")
        return redirect(url_for("result", analysis_id=analysis_id))

    try:
        report = analyze_career_fit(
            analysis["resume_text"], analysis["job_text"], analysis["job_title"], settings
        )
        update_analysis_report(DB_PATH, analysis_id, report)
    except Exception:
        app.logger.exception("AI review retry failed for analysis %s", analysis_id)
        flash("CareerPilot could not retry the AI review. Please try again.", "error")
        return redirect(url_for("result", analysis_id=analysis_id))

    if report.get("provider_issue"):
        flash("The AI review could not finish. See the note below for the reason.", "error")
    else:
        flash("The AI review is ready. Check the requirements and resume edits below.", "success")
    return redirect(url_for("result", analysis_id=analysis_id))


def _ensure_resume_draft(analysis: dict) -> None:
    """Fill new review fields for older saved analyses without changing their records."""
    report = analysis["report"]
    if report.get("resume_review") and report.get("tailored_resume"):
        return
    feedback = build_resume_feedback(
        analysis["resume_text"],
        analysis["job_text"],
        analysis["job_title"],
        report.get("matched_skills", []),
        [item.get("skill", "") for item in report.get("skill_gaps", []) if isinstance(item, dict)],
        report.get("evidence", []),
    )
    if not report.get("resume_review"):
        report["resume_review"] = feedback["resume_review"]
    existing = report.get("tailored_resume")
    if isinstance(existing, dict):
        report["tailored_resume"] = {**feedback["tailored_resume"], **existing}
        report["tailoring_method"] = report.get("tailoring_method", feedback["tailoring_method"])
    else:
        report["tailored_resume"] = feedback["tailored_resume"]
        report["tailoring_method"] = feedback["tailoring_method"]


@app.route("/analysis/<int:analysis_id>/tailored-resume.pdf", methods=["GET", "POST"])
def download_tailored_resume(analysis_id: int):
    analysis = get_analysis(DB_PATH, analysis_id)
    if analysis is None:
        abort(404)
    _ensure_resume_draft(analysis)
    if request.method == "POST":
        report = dict(analysis["report"])
        draft = dict(report.get("tailored_resume", {}))
        draft["summary"] = request.form.get("summary", "").strip()[:1200]
        draft["skills"] = [
            skill.strip()[:60]
            for skill in re.split(r"[,;\n]", request.form.get("skills", ""))
            if skill.strip()
        ][:20]
        draft["highlights"] = [
            line.strip()[:600]
            for line in request.form.get("highlights", "").splitlines()
            if line.strip()
        ][:8]
        approved_rewrites = []
        for index, rewrite in enumerate(draft.get("bullet_rewrites", [])):
            if not isinstance(rewrite, dict):
                continue
            tailored = request.form.get(f"rewritten_bullet_{index}", "").strip()[:600]
            if tailored:
                approved_rewrites.append({
                    "source": str(rewrite.get("source", ""))[:600],
                    "suggestion": tailored,
                })
        draft["approved_rewrites"] = approved_rewrites
        draft["user_approved"] = True
        report["tailored_resume"] = draft
        analysis = {**analysis, "report": report}
    buffer = make_resume_pdf(analysis)
    filename = secure_filename(f"{analysis['job_title']}-tailored-resume") or "tailored-resume"
    response = send_file(
        buffer,
        as_attachment=True,
        download_name=f"{filename}.pdf",
        mimetype="application/pdf",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/analysis/<int:analysis_id>/delete")
def confirm_delete(analysis_id: int):
    analysis = get_analysis(DB_PATH, analysis_id)
    if analysis is None:
        abort(404)
    return render_template("confirm_delete.html", analysis=analysis)


@app.post("/analysis/<int:analysis_id>/delete")
def remove_analysis(analysis_id: int):
    if not delete_analysis(DB_PATH, analysis_id):
        abort(404)
    flash("Analysis deleted. Its saved resume and job description were removed too.", "success")
    return redirect(url_for("history"))


@app.get("/history")
def history():
    return render_template("history.html", analyses=_recent_analyses(limit=100))


@app.errorhandler(413)
def too_large(_error):
    flash("That file is larger than 8 MB. Try a smaller resume file.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=os.environ.get("FLASK_DEBUG") == "1")
