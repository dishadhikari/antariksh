import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class ExperimentLogger:

    def __init__(
        self,
        database_path: str = "data/logs/experiment.db",
    ):
        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.database_path
        )

        self._create_table()

    def _create_table(self):

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                expected_step TEXT,
                detected_step TEXT NOT NULL,
                confidence REAL,
                status TEXT NOT NULL,
                message TEXT
            )
            """
        )

        self.connection.commit()

    def log_event(
        self,
        experiment_id: str,
        expected_step: str,
        detected_step: str,
        confidence: float,
        status: str,
        message: str = "",
    ):

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        self.connection.execute(
            """
            INSERT INTO events (
                timestamp,
                experiment_id,
                expected_step,
                detected_step,
                confidence,
                status,
                message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                experiment_id,
                expected_step,
                detected_step,
                float(confidence),
                status,
                message,
            ),
        )

        self.connection.commit()

    def get_events(
        self,
        experiment_id: str | None = None,
    ):

        if experiment_id is None:

            cursor = self.connection.execute(
                """
                SELECT
                    id,
                    timestamp,
                    experiment_id,
                    expected_step,
                    detected_step,
                    confidence,
                    status,
                    message
                FROM events
                ORDER BY id
                """
            )

        else:

            cursor = self.connection.execute(
                """
                SELECT
                    id,
                    timestamp,
                    experiment_id,
                    expected_step,
                    detected_step,
                    confidence,
                    status,
                    message
                FROM events
                WHERE experiment_id = ?
                ORDER BY id
                """,
                (experiment_id,),
            )

        columns = [
            "id",
            "timestamp",
            "experiment_id",
            "expected_step",
            "detected_step",
            "confidence",
            "status",
            "message",
        ]

        return [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]

    def export_json(
        self,
        output_path: str,
        experiment_id: str | None = None,
    ):

        events = self.get_events(
            experiment_id
        )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                events,
                file,
                indent=2,
            )

    def close(self):

        if self.connection:
            self.connection.close()
            self.connection = None