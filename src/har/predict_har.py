import json
from pathlib import Path

import joblib
import numpy as np

MODEL_FILE = Path("models/har/har_random_forest.joblib")
INPUT_FILE = Path("data/sequences/labeled/val_labeled.json")


def frame_features(frame):
    features = []
    features.extend(frame["keypoints"])
    features.extend(frame["keypoint_scores"])
    features.extend(frame["blue_box"])
    features.extend(frame["black_box"])
    return features


def sequence_features(sequence):
    features = []

    for frame in sequence["frames"]:
        features.extend(frame_features(frame))

    return features


def main():
    print("Loading HAR model...")
    model = joblib.load(MODEL_FILE)

    with open(INPUT_FILE, "r") as f:
        sequences = json.load(f)

    X = np.array(
        [sequence_features(sequence) for sequence in sequences],
        dtype=np.float32,
    )

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    print(f"Sequences: {len(sequences)}")
    print()

    for i, (sequence, prediction, probability) in enumerate(
        zip(sequences, predictions, probabilities),
        start=1,
    ):
        confidence = float(np.max(probability))

        print(
            f"{i:02d}. "
            f"{sequence['start_frame']} -> {sequence['end_frame']} | "
            f"Actual: {sequence['label']} | "
            f"Predicted: {prediction} | "
            f"Confidence: {confidence:.2f}"
        )


if __name__ == "__main__":
    main()