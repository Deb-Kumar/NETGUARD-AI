"""Parameterized SQLite storage for NETGUARD AI prediction history."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import ProjectPaths, ensure_directories, get_paths

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    prediction_type TEXT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    probability REAL,
    record_reference TEXT NOT NULL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_prediction_timestamp ON prediction_history(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_prediction_class ON prediction_history(predicted_class);
"""

_FORMULA_PREFIXES = ("=", "+", "@", "\t", "\r")


def sanitize_csv_cell(value: Any) -> Any:
    """Neutralize spreadsheet formula injection in exported string cells."""
    if not isinstance(value, str) or not value:
        return value
    stripped = value.lstrip()
    if value.startswith(_FORMULA_PREFIXES) or stripped.startswith(_FORMULA_PREFIXES):
        return "'" + value
    if value.startswith("-"):
        try:
            float(value)
        except ValueError:
            return "'" + value
    return value


def sanitize_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with object/string columns sanitized against CSV formula injection."""
    safe_df = df.copy()
    for col in safe_df.columns:
        if pd.api.types.is_string_dtype(safe_df[col]) or pd.api.types.is_object_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].map(sanitize_csv_cell)
    return safe_df


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with safe row factory and directory initialization."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(paths: ProjectPaths | None = None) -> Path:
    """Initialize the prediction history SQLite database schema."""
    resolved_paths = ensure_directories(paths or get_paths())
    with closing(_connect(resolved_paths.db_path)) as conn:
        with conn:
            conn.executescript(_SCHEMA_SQL)
    return resolved_paths.db_path


def log_prediction(
    *,
    prediction_type: str,
    model_name: str,
    predicted_class: str,
    probability: float | None,
    record_reference: str,
    notes: str = "",
    timestamp_utc: str | None = None,
    paths: ProjectPaths | None = None,
) -> int:
    """Insert one prediction summary record using parameterized SQL."""
    db_path = init_db(paths=paths)
    ts = timestamp_utc or datetime.now(timezone.utc).isoformat()
    safe_ref = str(record_reference)[:200]
    safe_notes = str(notes)[:400]

    with closing(_connect(db_path)) as conn:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO prediction_history (
                    timestamp_utc, prediction_type, model_name,
                    predicted_class, probability, record_reference, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    str(prediction_type),
                    str(model_name),
                    str(predicted_class),
                    float(probability) if probability is not None else None,
                    safe_ref,
                    safe_notes,
                ),
            )
            return int(cursor.lastrowid)


def query_history(
    *,
    search_text: str | None = None,
    prediction_type: str | None = None,
    predicted_class: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 500,
    paths: ProjectPaths | None = None,
) -> pd.DataFrame:
    """Query prediction history using safe parameterized SQL filters."""
    db_path = init_db(paths=paths)
    clauses: list[str] = []
    params: list[Any] = []

    if search_text:
        like_val = f"%{search_text.strip()}%"
        clauses.append("(record_reference LIKE ? OR model_name LIKE ? OR notes LIKE ?)")
        params.extend([like_val, like_val, like_val])
    if prediction_type and prediction_type != "All":
        clauses.append("prediction_type = ?")
        params.append(prediction_type)
    if predicted_class and predicted_class != "All":
        clauses.append("predicted_class = ?")
        params.append(predicted_class)
    if start_date:
        clauses.append("substr(timestamp_utc, 1, 10) >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("substr(timestamp_utc, 1, 10) <= ?")
        params.append(end_date)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT id, timestamp_utc, prediction_type, model_name,
               predicted_class, probability, record_reference, notes
        FROM prediction_history
        {where_sql}
        ORDER BY id DESC
        LIMIT ?
    """
    params.append(int(max(1, limit)))

    with closing(_connect(db_path)) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def export_history_csv(
    *,
    destination: Path | None = None,
    paths: ProjectPaths | None = None,
    **query_filters: Any,
) -> str | Path:
    """Export filtered prediction history as formula-safe CSV text or file."""
    df = sanitize_dataframe_for_csv(query_history(paths=paths, **query_filters))
    csv_text = df.to_csv(index=False)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(csv_text, encoding="utf-8")
        return destination
    return csv_text


def clear_history(*, confirm: bool = False, paths: ProjectPaths | None = None) -> int:
    """Delete all stored prediction history rows only when explicitly confirmed."""
    if not confirm:
        raise ValueError("Refusing to clear prediction history without confirm=True.")
    db_path = init_db(paths=paths)
    with closing(_connect(db_path)) as conn:
        with conn:
            cursor = conn.execute("DELETE FROM prediction_history")
            return int(cursor.rowcount)
