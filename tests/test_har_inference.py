from pathlib import Path

import numpy as np
import torch

from src.har.model import TCN
from src.har.inference import HARInference


def main():

    model_dir = Path("models/har/test")
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = model_dir / "best_tcn.pt"

    # Create a small test checkpoint if it does not exist.
    if not model_path.exists():

        model = TCN(
            input_features=8,
            num_classes=4,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_features": 8,
                "num_classes": 4,
            },
            model_path,
        )

    label_map = {
        0: "IDLE",
        1: "PICK_RED",
        2: "PICK_YELLOW",
        3: "OPEN_BOX",
    }

    inference = HARInference(
        model_path=str(model_path),
        label_map=label_map,
        device="cpu",
    )

    features = np.random.randn(
        30,
        8,
    ).astype("float32")

    result = inference.predict(features)

    print("Prediction:")
    print(result)

    assert "action" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

    print()
    print("HAR inference test passed.")


if __name__ == "__main__":
    main()