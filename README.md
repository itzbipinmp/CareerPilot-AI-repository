# CareerPilot AI

CareerPilot AI is a beginner-friendly job search assistant. It reads a resume and a job description, points out possible improvements, and prepares a separate role-focused resume as a clean, ATS-friendly PDF. It also creates a match report, skill-gap plan, and interview questions. The original resume stays unchanged, and the app never submits or sends the draft.

## Requirements and architecture

The first version is a small Flask app with three layers:

1. **Web interface (Flask + HTML/CSS):** upload a PDF, DOCX, or TXT resume and paste in a job description.
2. **Career assistant (Python + LangGraph):** extracts text, retrieves relevant resume passages using local TF-IDF search when available (with a simple keyword fallback), then runs skill-gap analysis, resume review and tailoring, and interview preparation. With Gemini or OpenAI configured, it compares job criteria with quoted resume evidence and verifies those quotations before showing a fit estimate. Local-only mode is a limited keyword scan and does not show an overall percentage. If LangGraph is not installed, the same tools run directly. The PDF keeps claims grounded in the uploaded resume and uses a one-column layout for readability and applicant tracking systems.
3. **Memory (SQLite):** saves each analysis, including the extracted resume and job text, so it can be viewed later in this app. Each saved item has a delete flow with a confirmation step; deleting removes its report and stored document text.

```text
Browser -> Flask routes -> resume retrieval -> LangGraph tool steps
                                      |       -> weighted evidence score
                                      |       -> optional structured LLM review
                                      +------> SQLite history

LangGraph steps: match role -> suggest skill gaps -> review and tailor resume -> prepare interview
```

The graph is deliberately simple and visible: each tool is one named step. Later, it can grow to add branching, user review, and more job-search tools.

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. CareerPilot has three analysis options:

- **Gemini/OpenAI cloud model:** Configure a key in the private server-side `.env` file. The browser never receives or submits the key. Resume text and job descriptions are sent to the selected provider when you request an analysis.
- **Local tools only:** Use keyword matching and resume retrieval without a language model. CareerPilot labels this as a limited scan and does not show an overall fit percentage.

With Gemini or OpenAI enabled, the review checks distinct required and preferred criteria against exact quotations from the resume. Only verified resume quotations count as evidence in that estimate. It remains a document comparison, not a prediction of interview or hiring success.

### Connect Gemini

1. Create a **new** Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey). Revoke the key previously shared in chat; do not reuse it.
2. In the project folder, open `.env` in a text editor. If it does not exist yet, copy `.env.example` and name the copy `.env`. Put your new key after `GEMINI_API_KEY=`. Do not paste the key into this chat or the browser page.
3. Save `.env`, then restart CareerPilot. It reads this private file when the server starts. The page should now say **Gemini is connected on this server**.
4. Upload your resume and paste the job description again. The browser never receives the API key. Resume text and the job description are sent to Google only when you request an analysis.

The `.env` file is excluded from Git by `.gitignore`. Keep it on your computer and never commit it. Google recommends restricting keys to the Gemini API and replacing a key if it is exposed; see [Google's API key security guide](https://ai.google.dev/gemini-api/docs/api-key). For a hosted app, use the hosting provider's private secret settings rather than uploading `.env`.

Environment configuration is also supported: `CAREERPILOT_PROVIDER` can be `gemini`, `openai`, or `local`. Use `GEMINI_API_KEY`/`GEMINI_MODEL` for Gemini and `OPENAI_API_KEY`/`OPENAI_MODEL` for OpenAI.

## Run with Docker

Docker keeps the app's Python dependencies separate from the rest of your computer. The database is stored in a named volume so analyses survive container restarts.

```bash
docker build -t careerpilot-ai .
docker volume create careerpilot-data
docker run --rm -p 5000:5000 \
  -e FLASK_SECRET_KEY='replace-with-a-long-random-secret' \
  -v careerpilot-data:/data \
  careerpilot-ai
```

Then open http://127.0.0.1:5000. For local use, add provider settings to `.env`. For a deployed container, provide API keys and `FLASK_SECRET_KEY` through your host's secret manager or a private env file excluded from Git; never place real values in this command or in source control. When enabled, the extracted resume text and job description are sent to the selected provider.

## Deployment outline

1. Put this project in a private Git repository.
2. Choose a hosting service that can build a Dockerfile, attach persistent storage, and restrict access to invited users.
3. Set `FLASK_SECRET_KEY` as a private environment variable. Add `OPENAI_API_KEY` or `GEMINI_API_KEY` only if you want model-written reviews and resume drafts.
4. Attach a persistent disk at `/data` and deploy behind private access controls.
5. Confirm a saved analysis remains after restarting the service. Back up the persistent disk because it contains resume text and reports.

This prototype does not include sign-in or per-user separation. Keep it on your own computer or behind private access controls. Do not make it public or share one instance across users until authentication, per-user data separation, and data deletion controls are in place.

## Documents and privacy

Supported resume formats: PDF, DOCX, and TXT. Files are processed in memory and their extracted text is saved with the analysis in `careerpilot.db`. The app binds to localhost by default. The PDF export is generated locally from the saved resume text and target role. If an OpenAI or Gemini key is configured, resume text and the job description are sent to the selected provider for model review; otherwise, analysis stays local. The app never sends the draft to an employer. Do not expose it publicly without adding authentication, per-user data separation, deletion controls, and a privacy review.

## Tests

Run `python -m unittest discover -s tests -v`.
