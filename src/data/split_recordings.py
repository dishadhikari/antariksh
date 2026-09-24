import json
import random
from pathlib import Path


class RecordingSplitter:

    def __init__(
        self,
        manifest_path="data/raw/box_experiment/metadata/manifest.json",
        output_path="data/splits.json",
        seed=42,
    ):
        self.manifest_path = Path(manifest_path)
        self.output_path = Path(output_path)
        self.seed = seed

    def split(self):

        with self.manifest_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            manifest = json.load(file)

        recordings = manifest.get("recordings", [])

        if not recordings:
            raise ValueError("No recordings found.")

        # Group recordings by execution type.
        groups = {}

        for recording in recordings:

            recording_id = recording["recording_id"]

            execution_type = recording[
                "execution"
            ]["type"]

            groups.setdefault(
                execution_type,
                []
            ).append(recording_id)

        rng = random.Random(self.seed)

        for group in groups.values():
            rng.shuffle(group)

        train_ids = []
        val_ids = []
        test_ids = []

        # Split each execution type separately.
        for execution_type, ids in groups.items():

            count = len(ids)

            if count == 1:
                # Very small category:
                # keep it in test for now.
                test_ids.extend(ids)

            elif count == 2:
                # One train, one test.
                train_ids.append(ids[0])
                test_ids.append(ids[1])

            else:
                train_count = max(
                    1,
                    round(count * 0.70)
                )

                val_count = max(
                    1,
                    round(count * 0.15)
                )

                if train_count + val_count >= count:
                    val_count = 1
                    train_count = count - 2

                train_ids.extend(
                    ids[:train_count]
                )

                val_ids.extend(
                    ids[
                        train_count:
                        train_count + val_count
                    ]
                )

                test_ids.extend(
                    ids[
                        train_count + val_count:
                    ]
                )

        rng.shuffle(train_ids)
        rng.shuffle(val_ids)
        rng.shuffle(test_ids)

        result = {
            "seed": self.seed,
            "method": (
                "recording_based_balanced_split"
            ),
            "execution_types": groups,
            "recordings": {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            }
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
                result,
                file,
                indent=2
            )

        print(
            f"Train recordings: {len(train_ids)}"
        )

        print(
            f"Validation recordings: {len(val_ids)}"
        )

        print(
            f"Test recordings: {len(test_ids)}"
        )

        print()
        print("Train:", train_ids)
        print("Val:", val_ids)
        print("Test:", test_ids)


if __name__ == "__main__":
    splitter = RecordingSplitter()
    splitter.split()