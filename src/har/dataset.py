from pathlib import Path
import json

import numpy as np


class HARDatasetBuilder:

    def __init__(
        self,
        sequence_length: int = 30,
        stride: int = 5,
    ):
        self.sequence_length = sequence_length
        self.stride = stride

    def create_sequences(
        self,
        features: np.ndarray,
        labels: list[str],
    ):
        """
        Convert frame-level features into temporal sequences.

        features:
            Shape: [num_frames, num_features]

        labels:
            One label per frame.
        """

        if len(features) != len(labels):
            raise ValueError(
                "Number of feature frames must match "
                "number of labels"
            )

        if len(features) < self.sequence_length:
            raise ValueError(
                "Not enough frames to create "
                "a sequence"
            )

        sequences = []
        sequence_labels = []

        for start in range(
            0,
            len(features) - self.sequence_length + 1,
            self.stride,
        ):

            end = (
                start +
                self.sequence_length
            )

            sequence = features[
                start:end
            ]

            window_labels = labels[
                start:end
            ]

            # Require the whole window to have
            # the same action label.
            if len(set(window_labels)) != 1:
                continue

            sequences.append(sequence)

            sequence_labels.append(
                window_labels[0]
            )

        if not sequences:
            raise ValueError(
                "No valid temporal sequences created"
            )

        return (
            np.asarray(
                sequences,
                dtype=np.float32
            ),
            sequence_labels,
        )

    def save(
        self,
        features: np.ndarray,
        labels: list[str],
        output_path: str,
    ):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savez_compressed(
            output_path,
            features=features,
            labels=np.asarray(
                labels
            )
        )

        print(
            f"Saved HAR dataset: {output_path}"
        )

        print(
            f"Features shape: {features.shape}"
        )

        print(
            f"Samples: {len(labels)}"
        )

    def load(
        self,
        input_path: str,
    ):

        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"HAR dataset not found: {input_path}"
            )

        data = np.load(
            input_path,
            allow_pickle=True
        )

        features = data["features"]
        labels = data["labels"].tolist()

        return features, labels