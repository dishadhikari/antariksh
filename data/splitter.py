import json
import random
from pathlib import Path


class DatasetSplitter:

    def __init__(
        self,
        manifest_path: str = "data/raw/box_experiment/metadata/manifest.json",
        output_path: str = "data/splits.json",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.manifest_path = Path(manifest_path)
        self.output_path = Path(output_path)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        total = train_ratio + val_ratio + test_ratio

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Train, validation and test ratios must sum to 1.0"
            )

    def load_manifest(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    def split(self):

        manifest = self.load_manifest()

        recordings = manifest.get("recordings", [])

        if not recordings:
            raise ValueError(
                "No recordings found in manifest"
            )

        # Group recordings by participant.
        # This prevents the same participant from appearing
        # in both training and testing.
        participants = {}

        for recording in recordings:

            recording_id = recording.get("recording_id")
            participant_id = recording.get("participant_id")

            if not recording_id:
                raise ValueError(
                    "Recording is missing recording_id"
                )

            if not participant_id:
                raise ValueError(
                    f"Recording {recording_id} "
                    "is missing participant_id"
                )

            participants.setdefault(
                participant_id,
                []
            ).append(recording_id)

        participant_ids = list(participants.keys())

        random.Random(self.seed).shuffle(
            participant_ids
        )

        total_participants = len(participant_ids)

        train_count = max(
            1,
            round(
                total_participants *
                self.train_ratio
            )
        )

        val_count = round(
            total_participants *
            self.val_ratio
        )

        # Make sure we don't exceed the total.
        if train_count + val_count > total_participants:
            val_count = max(
                0,
                total_participants - train_count
            )

        train_participants = participant_ids[
            :train_count
        ]

        val_participants = participant_ids[
            train_count:
            train_count + val_count
        ]

        test_participants = participant_ids[
            train_count + val_count:
        ]

        splits = {
            "train": [],
            "val": [],
            "test": []
        }

        for participant_id in train_participants:
            splits["train"].extend(
                participants[participant_id]
            )

        for participant_id in val_participants:
            splits["val"].extend(
                participants[participant_id]
            )

        for participant_id in test_participants:
            splits["test"].extend(
                participants[participant_id]
            )

        output = {
            "seed": self.seed,
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio
            },
            "participants": {
                "train": train_participants,
                "val": val_participants,
                "test": test_participants
            },
            "recordings": splits
        }

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.output_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                output,
                file,
                indent=2
            )

        return output