from pathlib import Path

import yaml


class ExperimentConfig:

    def __init__(self, path: str):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Experiment config not found: {self.path}"
            )

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.config = yaml.safe_load(file)

        if not self.config:
            raise ValueError(
                "Experiment config is empty"
            )

        self._validate()

    def _validate(self):

        if "experiment" not in self.config:
            raise ValueError(
                "Missing 'experiment' section"
            )

        if "sequence" not in self.config:
            raise ValueError(
                "Missing 'sequence' section"
            )

        sequence = self.config["sequence"]

        if not isinstance(sequence, list):
            raise ValueError(
                "'sequence' must be a list"
            )

        if not sequence:
            raise ValueError(
                "Experiment sequence cannot be empty"
            )

        if len(sequence) != len(set(sequence)):
            raise ValueError(
                "Experiment sequence contains duplicates"
            )

    @property
    def experiment_id(self) -> str:
        return self.config[
            "experiment"
        ]["id"]

    @property
    def experiment_name(self) -> str:
        return self.config[
            "experiment"
        ]["name"]

    @property
    def sequence(self) -> list[str]:
        return self.config["sequence"]

    @property
    def confidence_threshold(self) -> float:
        return self.config.get(
            "settings",
            {},
        ).get(
            "confidence_threshold",
            0.60,
        )

    @property
    def temporal_window(self) -> int:
        return self.config.get(
            "settings",
            {},
        ).get(
            "temporal_window",
            5,
        )

    @property
    def minimum_votes(self) -> int:
        return self.config.get(
            "settings",
            {},
        ).get(
            "minimum_votes",
            3,
        )