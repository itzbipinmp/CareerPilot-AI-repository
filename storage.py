"""SQLite persistence for CareerPilot analyses."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(db_path)) as connection:
        with connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_title TEXT NOT NULL,
                    resume_name TEXT NOT NULL,
                    resume_text TEXT NOT NULL,
                    job_text TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )


def save_analysis(db_path: Path, **values) -> int:
    with closing(connect(db_path)) as connection:
        with connection:
            cursor = connection.execute(
                """INSERT INTO analyses
                   (job_title, resume_name, resume_text, job_text, report_json, created_at)
                   VALUES (:job_title, :resume_name, :resume_text, :job_text, :report_json, :created_at)""",
                {**values, "report_json": json.dumps(values["report"], ensure_ascii=False)},
            )
            return int(cursor.lastrowid)


def list_analyses(db_path: Path, limit: int = 20) -> list[dict]:
    with closing(connect(db_path)) as connection:
        rows = connection.execute(
            "SELECT id, job_title, resume_name, created_at, report_json FROM analyses ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{**dict(row), "report": json.loads(row["report_json"])} for row in rows]


def get_analysis(db_path: Path, analysis_id: int) -> dict | None:
    with closing(connect(db_path)) as connection:
        row = connection.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if row is None:
        return None
    record = dict(row)
    record["report"] = json.loads(record.pop("report_json"))
    return record


def update_analysis_report(db_path: Path, analysis_id: int, report: dict) -> None:
    """Save refreshed analysis fields without modifying the stored source documents."""
    with closing(connect(db_path)) as connection:
        with connection:
            connection.execute(
                "UPDATE analyses SET report_json = ? WHERE id = ?",
                (json.dumps(report, ensure_ascii=False), analysis_id),
            )


def delete_analysis(db_path: Path, analysis_id: int) -> bool:
    """Delete one analysis and the resume/job text saved with it."""
    with closing(connect(db_path)) as connection:
        with connection:
            cursor = connection.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
            return cursor.rowcount > 0
