import json
import shutil
from pathlib import Path


class DetectionDatasetPreparer:

    def __init__(
        self,
        manifest_path="data/raw/box_experiment/metadata/manifest.json",
        splits_path="data/splits.json",
        frames_dir="data/detection/images/raw",
        output_dir="data/detection",
    ):
        self.manifest_path = Path(manifest_path)
        self.splits_path = Path(splits_path)
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)

    def load_json(self, path):

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def create_directories(self):

        for split in [
            "train",
            "val",
            "test",
        ]:

            (
                self.output_dir
                / "images"
                / split
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                self.output_dir
                / "labels"
                / split
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    def prepare(self):

        manifest = self.load_json(
            self.manifest_path
        )

        splits = self.load_json(
            self.splits_path
        )

        self.create_directories()

        recordings = {
            item["recording_id"]: item
            for item in manifest.get(
                "recordings",
                []
            )
        }

        copied = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        for split in [
            "train",
            "val",
            "test",
        ]:

            recording_ids = splits[
                "recordings"
            ].get(
                split,
                []
            )

            for recording_id in recording_ids:

                if recording_id not in recordings:
                    raise ValueError(
                        f"Recording {recording_id} "
                        "not found in manifest"
                    )

                recording = recordings[
                    recording_id
                ]

                video_path = Path(
                    recording["video"]["path"]
                )

                video_name = video_path.stem

                pattern = f"{video_name}_*.jpg"

                frames = sorted(
                    self.frames_dir.glob(pattern)
                )

                print(
                    f"{recording_id} "
                    f"({video_name}.mp4) -> "
                    f"{split}: "
                    f"{len(frames)} frames"
                )

                for frame in frames:

                    destination = (
                        self.output_dir
                        / "images"
                        / split
                        / frame.name
                    )

                    shutil.copy2(
                        frame,
                        destination,
                    )

                    copied[split] += 1

        print()
        print(
            f"Train images: {copied['train']}"
        )

        print(
            f"Val images: {copied['val']}"
        )

        print(
            f"Test images: {copied['test']}"
        )

        print(
            "Detection dataset preparation complete."
        )


if __name__ == "__main__":

    preparer = DetectionDatasetPreparer()
    preparer.prepare()