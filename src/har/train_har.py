import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


TRAIN_FILE = Path("data/sequences/labeled/train_labeled.json")
VAL_FILE = Path("data/sequences/labeled/val_labeled.json")

MODEL_DIR = Path("models/har")
MODEL_FILE = MODEL_DIR / "har_random_forest.joblib"


CLASS_NAMES = [
    "LEFT_HAND_UP",
    "RIGHT_HAND_UP",
    "PICK_BLUE_BOX",
    "PUT_DOWN_BLUE_BOX",
    "PICK_BLACK_BOX",
    "PUT_DOWN_BLACK_BOX",
]


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


def load_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)

    X = []
    y = []

    for sequence in data:
        X.append(sequence_features(sequence))
        y.append(sequence["label"])

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_dataset(TRAIN_FILE)
    X_val, y_val = load_dataset(VAL_FILE)

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Feature size:", X_train.shape[1])

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    print("\nTraining HAR model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    print(f"\nValidation accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=CLASS_NAMES,
            zero_division=0,
        )
    )

    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()