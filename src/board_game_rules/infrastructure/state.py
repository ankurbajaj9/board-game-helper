import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from board_game_rules.domain.models import ParseReport, Rulebook


class IndexState:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS active_documents (
                    document_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    document TEXT NOT NULL,
                    report TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ingestion_attempts (
                    path TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS document_title_overrides (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL
                );
            """)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def active(self, document_id: str) -> tuple[Rulebook, ParseReport, str] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT document, report, collection FROM active_documents WHERE document_id=?",
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        return (
            Rulebook.model_validate_json(row["document"]),
            ParseReport.model_validate_json(row["report"]),
            row["collection"],
        )

    def documents(self) -> list[Rulebook]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT document FROM active_documents ORDER BY document_id"
            ).fetchall()
        return [Rulebook.model_validate_json(row["document"]) for row in rows]

    def title_overrides(self) -> dict[str, str]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT document_id, title FROM document_title_overrides"
            ).fetchall()
        return {row["document_id"]: row["title"] for row in rows}

    def set_title_override(self, document_id: str, title: str) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO document_title_overrides (document_id, title) VALUES (?, ?)
                ON CONFLICT(document_id) DO UPDATE SET title=excluded.title""",
                (document_id, title),
            )

    def publish(self, doc: Rulebook, report: ParseReport, collection: str) -> None:
        if report.document_id != doc.document_id:
            raise ValueError("Cannot activate a report for a different document.")
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO active_documents VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    fingerprint=excluded.fingerprint, collection=excluded.collection,
                    document=excluded.document, report=excluded.report""",
                (
                    doc.document_id,
                    report.fingerprint,
                    collection,
                    doc.model_dump_json(),
                    report.model_dump_json(),
                ),
            )
            self._attempt(connection, doc.local_path, "ready", None)

    def attempt(self, path: str, status: str, error: str | None = None) -> None:
        with self.connection() as connection:
            self._attempt(connection, path, status, error)

    @staticmethod
    def _attempt(connection: sqlite3.Connection, path: str, status: str, error: str | None) -> None:
        connection.execute(
            """INSERT INTO ingestion_attempts (path, status, error) VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET status=excluded.status, error=excluded.error,
                updated_at=CURRENT_TIMESTAMP""",
            (path, status, error),
        )

    def attempts(self) -> list[dict[str, str | None]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT path, status, error, updated_at FROM ingestion_attempts ORDER BY path"
            ).fetchall()
        return [dict(row) for row in rows]
