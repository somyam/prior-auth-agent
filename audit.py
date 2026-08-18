"""SQLite-backed audit log for prior authorization decisions."""
import os
import sqlite3

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
    response_time_seconds REAL
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def log_decision(result: dict) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """INSERT INTO prior_auth_audit_log
               (patient_id, diagnosis_code, procedure, decision, reasoning, timestamp, response_time_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                result["patient_id"],
                result["diagnosis_code"],
                result["procedure"],
                result["decision"],
                result["reasoning"],
                result["timestamp"],
                result["response_time_seconds"],
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
