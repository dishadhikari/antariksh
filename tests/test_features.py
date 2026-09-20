import numpy as np

from src.har.features import HARFeatureExtractor
from src.interaction.interaction_engine import (
    InteractionFeatures,
)


def main():

    interaction_features = InteractionFeatures(

        left_hand_red_distance=100.0,
        right_hand_red_distance=50.0,

        left_hand_yellow_distance=200.0,
        right_hand_yellow_distance=150.0,

        left_hand_experiment_box_distance=300.0,
        right_hand_experiment_box_distance=250.0,

        left_hand_speed=20.0,
        right_hand_speed=30.0,
    )

    extractor = HARFeatureExtractor()

    vector = extractor.extract(
        interaction_features
    )

    normalized = extractor.normalize(
        vector,
        width=640,
        height=480
    )

    print("Feature names:")
    print(extractor.FEATURE_NAMES)

    print("\nRaw vector:")
    print(vector)

    print("\nNormalized vector:")
    print(normalized)

    print(
        "\nFeature count:",
        extractor.feature_count
    )

    assert isinstance(
        vector,
        np.ndarray
    )

    assert vector.shape == (8,)

    assert normalized.shape == (8,)


if __name__ == "__main__":
    main()