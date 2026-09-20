from dataclasses import asdict
from typing import Any

import numpy as np

from src.interaction.interaction_engine import InteractionFeatures


class HARFeatureExtractor:

    FEATURE_NAMES = [
        "left_hand_red_distance",
        "right_hand_red_distance",

        "left_hand_yellow_distance",
        "right_hand_yellow_distance",

        "left_hand_experiment_box_distance",
        "right_hand_experiment_box_distance",

        "left_hand_speed",
        "right_hand_speed",
    ]

    def __init__(self):
        self.feature_count = len(
            self.FEATURE_NAMES
        )

    def extract(
        self,
        interaction_features: InteractionFeatures,
    ) -> np.ndarray:

        values = asdict(
            interaction_features
        )

        vector = np.array(
            [
                values[name]
                for name in self.FEATURE_NAMES
            ],
            dtype=np.float32
        )

        return vector

    def extract_dict(
        self,
        interaction_features: InteractionFeatures,
    ) -> dict[str, float]:

        values = asdict(
            interaction_features
        )

        return {
            name: float(values[name])
            for name in self.FEATURE_NAMES
        }

    def normalize(
        self,
        vector: np.ndarray,
        width: float,
        height: float,
    ) -> np.ndarray:

        """
        Normalize distance-like features by image diagonal.

        Speeds are also normalized by the same scale.
        """

        if width <= 0 or height <= 0:
            raise ValueError(
                "Image width and height must be positive"
            )

        diagonal = np.sqrt(
            width ** 2 +
            height ** 2
        )

        if diagonal == 0:
            raise ValueError(
                "Image diagonal cannot be zero"
            )

        normalized = vector.copy()

        normalized /= diagonal

        return normalized