import json
from pathlib import Path

import joblib
import numpy as np

from src.har.experiment_evaluator import ExperimentEvaluator


MODEL_FILE = Path("models/har/har_random_forest.joblib")
FEATURE_DIR = Path("data/features/val")

RECORDING_ID = "5"


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

    files = sorted(
        FEATURE_DIR.glob(f"{RECORDING_ID}_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    print(f"Recording: {RECORDING_ID}")
    print(f"Feature files: {len(files)}")

    evaluator = ExperimentEvaluator()

    predictions = []

    for file in files:
        with open(file, "r") as f:
            frame = json.load(f)

        # Use one frame as a temporary sequence.
        # The actual temporal sequence is handled below.
        predictions.append(frame)

    # Build 5-frame sliding windows.
    WINDOW_SIZE = 5

    for i in range(len(predictions) - WINDOW_SIZE + 1):
        window = {
            "frames": predictions[i:i + WINDOW_SIZE]
        }

        X = np.array(
            [sequence_features(window)],
            dtype=np.float32,
        )

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        confidence = float(np.max(probability))

        result = evaluator.update(prediction, confidence)

        if result["confirmed_action"]:
            print(
                f"{i + WINDOW_SIZE:03d} | "
                f"Prediction: {prediction:<22} | "
                f"Confidence: {confidence:.2f} | "
                f"Confirmed: {result['confirmed_action']:<22} | "
                f"Status: {result['status']}"
            )

        if evaluator.state_machine.status in {
            "COMPLETE",
            "SKIP_ORDER",
        }:
            break

    print("\n==============================")
    print("Detected sequence:")
    for action in evaluator.state_machine.completed:
        print(f"  {action}")

    print(
        f"\nFinal experiment status: "
        f"{evaluator.state_machine.status}"
    )


if __name__ == "__main__":
    main()