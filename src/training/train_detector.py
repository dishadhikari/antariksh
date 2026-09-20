from pathlib import Path

from ultralytics import YOLO


DATASET_CONFIG = "data/detection/dataset.yaml"
BASE_MODEL = "yolo11n.pt"

OUTPUT_DIR = "models/detection"


def train_detector():

    dataset_path = Path(DATASET_CONFIG)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {dataset_path}"
        )

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading YOLO model...")

    model = YOLO(BASE_MODEL)

    print("Starting custom detector training...")

    results = model.train(
        data=str(dataset_path),

        # Start small for development.
        epochs=50,
        imgsz=640,
        batch=8,

        # CPU development.
        device="cpu",

        # Dataset/output settings.
        project=OUTPUT_DIR,
        name="box_detector",

        # Reproducibility.
        seed=42,

        # Save checkpoints.
        save=True,

        # Validate during training.
        val=True,

        # Use pretrained weights.
        pretrained=True,

        # Early stopping.
        patience=10,

        verbose=True,
    )

    print()
    print("Training completed.")

    print(
        f"Best model should be located at:"
        f" {OUTPUT_DIR}/box_detector/weights/best.pt"
    )

    return results


if __name__ == "__main__":
    train_detector()