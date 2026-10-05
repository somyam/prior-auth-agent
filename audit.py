"""SQLite-backed audit log for prior authorization decisions."""
import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "audit_log.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS prior_auth_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    diagnosis_code TEXT,
    procedure TEXT,
    decision TEXT,
    reasoning TEXT,
    timestamp TEXT,
    response_time_seconds REAL,
    tool_trace TEXT,
    assessment TEXT,
    policy_evidence TEXT
)
"""

REVIEW_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_review (
    thread_id TEXT PRIMARY KEY,
    patient_id TEXT,
    diagnosis_code TEXT,
    procedure TEXT,
    reasoning TEXT,
    queued_at TEXT
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.execute(REVIEW_SCHEMA)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(prior_auth_audit_log)")}
    for name in ("tool_trace", "assessment", "policy_evidence"):
        if name not in existing:
            conn.execute(f"ALTER TABLE prior_auth_audit_log ADD COLUMN {name} TEXT")
    return conn


def log_decision(result: dict) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """INSERT INTO prior_auth_audit_log
               (patient_id, diagnosis_code, procedure, decision, reasoning, timestamp, response_time_seconds,
                tool_trace, assessment, policy_evidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result["patient_id"],
                result["diagnosis_code"],
                result["procedure"],
                result["decision"],
                result["reasoning"],
                result["timestamp"],
                result["response_time_seconds"],
                json.dumps(result.get("tool_trace", [])),
                json.dumps(result.get("assessment", {})),
                result.get("policy_info", ""),
            ),
        )
    conn.close()


def fetch_all() -> list[tuple]:
    conn = _connect()
    rows = conn.execute(
        """SELECT patient_id, diagnosis_code, procedure, decision, timestamp, response_time_seconds
           FROM prior_auth_audit_log ORDER BY id DESC"""
    ).fetchall()
    conn.close()
    return rows


def queue_for_review(thread_id: str, patient_id: str, diagnosis_code: str, procedure: str, reasoning: str) -> None:
    """Record a PENDED case as awaiting human review, keyed by its LangGraph thread_id."""
    conn = _connect()
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO pending_review
               (thread_id, patient_id, diagnosis_code, procedure, reasoning, queued_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (thread_id, patient_id, diagnosis_code, procedure, reasoning, datetime.now().isoformat()),
        )
    conn.close()


def fetch_pending_reviews() -> list[tuple]:
    conn = _connect()
    rows = conn.execute(
        """SELECT thread_id, patient_id, diagnosis_code, procedure, reasoning, queued_at
           FROM pending_review ORDER BY queued_at"""
    ).fetchall()
    conn.close()
    return rows


def resolve_review(thread_id: str) -> None:
    """Remove a case from the review queue once a reviewer has resumed its graph."""
    conn = _connect()
    with conn:
        conn.execute("DELETE FROM pending_review WHERE thread_id = ?", (thread_id,))
    conn.close()
