from src.har.features import HARFeatureExtractor
from src.interaction.interaction_engine import InteractionFeatures


def main():

    features = InteractionFeatures(
        left_hand_red_distance=100.0,
        right_hand_red_distance=150.0,
        left_hand_yellow_distance=200.0,
        right_hand_yellow_distance=250.0,
        left_hand_experiment_box_distance=300.0,
        right_hand_experiment_box_distance=350.0,
        left_hand_speed=10.0,
        right_hand_speed=15.0,
    )

    extractor = HARFeatureExtractor()

    vector = extractor.extract(features)

    print("HAR feature vector:")
    print(vector)

    print()
    print("Shape:", vector.shape)
    print("Feature count:", extractor.feature_count)

    assert vector.shape == (8,)
    assert extractor.feature_count == 8

    print()
    print("HAR feature extraction passed.")


if __name__ == "__main__":
    main()