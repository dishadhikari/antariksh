from collections import deque

import numpy as np


class TemporalFeatureBuffer:

    def __init__(
        self,
        sequence_length: int = 30,
        feature_count: int = 8,
    ):
        if sequence_length <= 0:
            raise ValueError(
                "Sequence length must be positive"
            )

        if feature_count <= 0:
            raise ValueError(
                "Feature count must be positive"
            )

        self.sequence_length = sequence_length
        self.feature_count = feature_count

        self.buffer = deque(
            maxlen=sequence_length
        )

    def add(self, features):

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        if features.shape != (self.feature_count,):
            raise ValueError(
                f"Expected feature vector shape "
                f"({self.feature_count},), "
                f"got {features.shape}"
            )

        self.buffer.append(features)

    @property
    def is_ready(self) -> bool:
        return len(self.buffer) >= self.sequence_length

    @property
    def size(self) -> int:
        return len(self.buffer)

    def get_sequence(self) -> np.ndarray:

        if not self.is_ready:
            raise ValueError(
                "Temporal buffer is not ready"
            )

        return np.stack(
            list(self.buffer),
            axis=0,
        )

    def clear(self):
        self.buffer.clear()