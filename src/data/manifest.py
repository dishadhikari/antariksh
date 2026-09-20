import json
from pathlib import Path
from typing import Any


class DatasetManifest:

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def create(self, dataset_id: str, description: str = ""):
        data = {
            "dataset": {
                "id": dataset_id,
                "version": "1.0",
                "description": description
            },
            "recordings": []
        }

        self._save(data)
        return data

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.path}"
            )

        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def add_recording(self, recording: dict[str, Any]):
        data = self.load()

        if "recordings" not in data:
            data["recordings"] = []

        recording_id = recording.get("recording_id")

        if not recording_id:
            raise ValueError(
                "Recording must contain 'recording_id'"
            )

        existing_ids = {
            item["recording_id"]
            for item in data["recordings"]
        }

        if recording_id in existing_ids:
            raise ValueError(
                f"Recording already exists: {recording_id}"
            )

        data["recordings"].append(recording)

        self._save(data)

    def _save(self, data: dict[str, Any]):
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                data,
                file,
                indent=2
            )