import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as web_app
from pypdf import PdfReader
from resume_export import make_resume_pdf
from storage import init_db, save_analysis


class FlaskWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.previous_llm_settings = web_app.app.config["LLM_SETTINGS"]
        web_app.app.config["LLM_SETTINGS"] = {"provider": "local", "enabled": False, "api_key": "", "model": ""}

    def tearDown(self):
        web_app.app.config["LLM_SETTINGS"] = self.previous_llm_settings

    def test_env_file_loads_server_values_without_overriding_shell_values(self):
        with tempfile.TemporaryDirectory() as folder:
            env_file = Path(folder) / ".env"
            env_file.write_text("# private settings\nCAREERPILOT_TEST_KEY='file-value'\nCAREERPILOT_TEST_KEEP=file-value\n", encoding="utf-8")
            with patch.dict("os.environ", {"CAREERPILOT_TEST_KEEP": "shell-value"}, clear=False):
                web_app._load_env_file(env_file)
                import os
                self.assertEqual(os.environ["CAREERPILOT_TEST_KEY"], "file-value")
                self.assertEqual(os.environ["CAREERPILOT_TEST_KEEP"], "shell-value")
                os.environ.pop("CAREERPILOT_TEST_KEY", None)

    def test_api_key_is_not_collected_by_browser(self):
        client = web_app.app.test_client()
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertNotIn(b'name="api_key"', page.data)
        self.assertNotIn(b'id="api-key"', page.data)
        self.assertIn(b"private", page.data)
        self.assertIn(b".env", page.data)
        with client.session_transaction() as browser_session:
            csrf_token = browser_session["csrf_token"]
        self.assertEqual(client.post("/settings/ai", data={"csrf_token": csrf_token}).status_code, 404)

    def test_resume_pdf_keeps_experience_and_does_not_invent_analyst_facts(self):
        resume = """Asha Example
asha@example.com | Pune, India
TECHNICAL SKILLS
Microsoft Excel, SQL, Python
WORK EXPERIENCE
Reporting Intern | Student Services
Prepared weekly attendance reports and checked records for missing values.
PROJECTS
Student Attendance Dashboard
Built a Tableau dashboard to summarize attendance by course.
EDUCATION
Bachelor of Arts, 2024
"""
        pdf = make_resume_pdf({
            "resume_text": resume,
            "job_title": "Junior Data Analyst",
            "report": {
                "matched_skills": ["sql", "tableau"],
                "evidence": ["Prepared weekly attendance reports and checked records for missing values."],
                "tailored_resume": {
                    "summary": "BCA graduate with face recognition project experience.",
                    "highlights": ["Built a face recognition system with Python."],
                },
            },
        })
        reader = PdfReader(pdf)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.assertIn("EXPERIENCE", text)
        self.assertIn("Reporting Intern", text)
        self.assertIn("Student Attendance Dashboard", text)
        self.assertIn("Bachelor of Arts, 2024", text)
        self.assertNotIn("BCA", text)
        self.assertNotIn("face recognition", text.lower())
        skills_start = text.index("TECHNICAL SKILLS")
        self.assertLess(text.index("SQL", skills_start), text.index("Microsoft Excel", skills_start))

    def test_resume_pdf_uses_user_reviewed_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "resume-edit-test.db"
            init_db(database)
            with patch.object(web_app, "DB_PATH", database):
                analysis_id = save_analysis(
                    database,
                    job_title="Junior Data Analyst",
                    resume_name="resume.txt",
                    resume_text="Asha Example\nasha@example.com | Pune, India\nSKILLS\nExcel, SQL\nPROJECTS\nSales Report\nBuilt a weekly report in Excel.",
                    job_text="Use SQL and Excel to analyze records, build reports, and share findings with the team.",
                    report={
                        "matched_skills": ["excel", "sql"],
                        "tailored_resume": {
                            "headline": "Junior Data Analyst", "summary": "", "skills": ["sql", "excel"], "highlights": [],
                            "bullet_rewrites": [{
                                "source": "Built a weekly report in Excel.",
                                "suggestion": "Built weekly reports using Excel.",
                                "reason": "Shows reporting experience with a required tool.",
                            }],
                        },
                    },
                    created_at="2026-10-15T00:00:00+00:00",
                )
                client = web_app.app.test_client()
                client.get("/")
                with client.session_transaction() as browser_session:
                    csrf_token = browser_session["csrf_token"]
                response = client.post(
                    f"/analysis/{analysis_id}/tailored-resume.pdf",
                    data={
                        "csrf_token": csrf_token,
                        "summary": "Junior analyst with verified Excel reporting experience.",
                        "skills": "SQL, Excel",
                        "highlights": "Built a weekly report in Excel.",
                        "rewritten_bullet_0": "Created weekly Excel reports.",
                    },
                )
                self.assertEqual(response.status_code, 200)
                text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(response.data)).pages)
                self.assertIn("verified Excel reporting experience", text)
                self.assertIn("Created weekly Excel reports.", text)
                self.assertNotIn("Built a weekly report in Excel.", text)
                self.assertNotIn("CareerPilot AI", text)
                self.assertIn("Pune, India", text)

    def test_health_check(self):
        response = web_app.app.test_client().get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ok"})

    def test_retry_ai_review_reuses_saved_resume_and_job(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "retry-ai-test.db"
            init_db(database)
            with patch.object(web_app, "DB_PATH", database):
                analysis_id = save_analysis(
                    database,
                    job_title="Junior Data Analyst",
                    resume_name="resume.txt",
                    resume_text="Excel and SQL project experience.",
                    job_text="Junior analyst role using Excel and SQL to prepare reports for a team.",
                    report={"provider": "Local tools (Gemini unavailable)", "provider_issue": "Gemini busy", "score_version": 4, "score_breakdown": {}},
                    created_at="2026-10-15T00:00:00+00:00",
                )
                web_app.app.config["LLM_SETTINGS"] = {"provider": "gemini", "enabled": True, "api_key": "test-key-only", "model": "gemini-3.8-flash"}
                generated_report = {
                    "provider": "Gemini + RAG + local tools",
                    "requirements": [{"requirement": "SQL", "priority": "required", "evidence_label": "Strong evidence", "evidence_quote": "Excel and SQL project experience.", "reason": "The resume names SQL project work."}],
                    "match_score": 80,
                }
                with patch.object(web_app, "analyze_career_fit", return_value=generated_report) as analyze_mock:
                    client = web_app.app.test_client()
                    client.get("/")
                    with client.session_transaction() as browser_session:
                        csrf_token = browser_session["csrf_token"]
                    response = client.post(
                        f"/analysis/{analysis_id}/retry-ai",
                        data={"csrf_token": csrf_token},
                    )
                self.assertEqual(response.status_code, 302)
                analyze_mock.assert_called_once_with(
                    "Excel and SQL project experience.",
                    "Junior analyst role using Excel and SQL to prepare reports for a team.",
                    "Junior Data Analyst",
                    web_app.app.config["LLM_SETTINGS"],
                )
                stored = web_app.get_analysis(database, analysis_id)
                self.assertEqual(stored["report"], generated_report)

    def test_upload_creates_report_and_history(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "careerpilot-test.db"
            init_db(database)
            with patch.object(web_app, "DB_PATH", database):
                client = web_app.app.test_client()
                page = client.get("/")
                self.assertEqual(page.status_code, 200)
                with client.session_transaction() as browser_session:
                    csrf_token = browser_session["csrf_token"]
                response = client.post(
                    "/analyze",
                    data={
                        "csrf_token": csrf_token,
                        "job_title": "Junior Python Developer",
                        "job_description": "We are looking for a Python developer with Flask, SQL, AWS, and communication skills. Build APIs and work with our team.",
                        "resume": (io.BytesIO(b"Python developer with Flask and SQL project experience."), "resume.txt"),
                    },
                    content_type="multipart/form-data",
                )
                self.assertEqual(response.status_code, 302)
                result = client.get(response.headers["Location"])
                self.assertEqual(result.status_code, 200)
                self.assertIn(b"Junior Python Developer", result.data)
                self.assertIn(b"LangGraph + local tools", result.data)
                self.assertIn(b"What you could improve", result.data)
                self.assertIn(b"Your tailored resume", result.data)
                download = client.get(f"/analysis/{response.headers['Location'].rsplit('/', 1)[1]}/tailored-resume.pdf")
                self.assertEqual(download.status_code, 200)
                self.assertEqual(download.mimetype, "application/pdf")
                self.assertTrue(download.data.startswith(b"%PDF-"))
                draft = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(download.data)).pages)
                self.assertIn("JUNIOR PYTHON DEVELOPER", draft)
                self.assertIn("Python developer", draft)
                history = client.get("/history")
                self.assertIn(b"Junior Python Developer", history.data)

    def test_validation_error_keeps_job_description(self):
        client = web_app.app.test_client()
        client.get("/")
        with client.session_transaction() as browser_session:
            csrf_token = browser_session["csrf_token"]
        job_description = "A detailed role description that should stay visible after a missing resume error."
        response = client.post(
            "/analyze",
            data={"csrf_token": csrf_token, "job_title": "Python Developer", "job_description": job_description},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(job_description.encode(), response.data)
        self.assertIn(b"Choose a resume file to continue", response.data)

    def test_old_analysis_gets_review_and_draft_when_opened(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "legacy-test.db"
            init_db(database)
            with patch.object(web_app, "DB_PATH", database):
                analysis_id = save_analysis(
                    database,
                    job_title="Junior Data Analyst",
                    resume_name="resume.txt",
                    resume_text="Excel analyst who built a sales dashboard for a college project.",
                    job_text="Looking for Excel and SQL skills to build and explain analytics reports for clients.",
                    report={
                        "match_score": 50,
                        "summary": "An older saved report.",
                        "matched_skills": ["excel"],
                        "skill_gaps": [{"skill": "sql", "next_step": "Learn SQL."}],
                        "interview_questions": [],
                        "evidence": ["built a sales dashboard"],
                        "provider": "Local tools",
                    },
                    created_at="2026-10-15T00:00:00+00:00",
                )
                response = web_app.app.test_client().get(f"/analysis/{analysis_id}")
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"What you could improve", response.data)
                self.assertIn(b"Your tailored resume", response.data)
                refreshed = web_app.get_analysis(database, analysis_id)
                self.assertEqual(refreshed["report"]["score_version"], 4)
                self.assertEqual(refreshed["report"]["score_method"], "Local keyword estimate (limited evidence)")
                self.assertIn("score_breakdown", refreshed["report"])
                self.assertIn(b"AI needed", response.data)
                self.assertIn(b"No reliable percentage is available", response.data)
                self.assertNotIn(b"Required and preferred skills", response.data)

    def test_delete_requires_confirmation_then_removes_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "careerpilot-delete-test.db"
            init_db(database)
            with patch.object(web_app, "DB_PATH", database):
                client = web_app.app.test_client()
                client.get("/")
                with client.session_transaction() as browser_session:
                    csrf_token = browser_session["csrf_token"]
                created = client.post(
                    "/analyze",
                    data={
                        "csrf_token": csrf_token,
                        "job_title": "Junior Data Analyst",
                        "job_description": "Analyze data with SQL, Excel, Python, and dashboards. Communicate findings clearly and work with the team to support decisions.",
                        "resume": (io.BytesIO(b"Junior data analyst using SQL and Excel with a reporting project and careful documentation."), "resume.txt"),
                    },
                    content_type="multipart/form-data",
                )
                analysis_id = int(created.headers["Location"].rsplit("/", 1)[1])
                confirmation = client.get(f"/analysis/{analysis_id}/delete")
                self.assertEqual(confirmation.status_code, 200)
                self.assertIn("You can’t undo this.".encode(), confirmation.data)
                self.assertEqual(client.get(f"/analysis/{analysis_id}").status_code, 200)
                history = client.get("/history")
                self.assertIn(f"/analysis/{analysis_id}/delete".encode(), history.data)
                rejected = client.post(f"/analysis/{analysis_id}/delete")
                self.assertEqual(rejected.status_code, 400)
                deleted = client.post(f"/analysis/{analysis_id}/delete", data={"csrf_token": csrf_token})
                self.assertEqual(deleted.status_code, 302)
                self.assertEqual(client.get(f"/analysis/{analysis_id}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
