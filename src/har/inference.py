from pathlib import Path

import numpy as np
import torch

from src.har.model import TCN


class HARInference:

    def __init__(
        self,
        model_path: str,
        label_map: dict[int, str],
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.label_map = label_map

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"HAR model not found: {model_path}"
            )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model = TCN(
            input_features=checkpoint["input_features"],
            num_classes=checkpoint["num_classes"],
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

    def predict(self, features: np.ndarray):
        if features.ndim != 2:
            raise ValueError(
                "Expected features with shape "
                "[sequence_length, feature_count]"
            )

        tensor = torch.tensor(
            features,
            dtype=torch.float32,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, class_index = (
                probabilities.max(dim=1)
            )

        class_index = class_index.item()
        confidence = confidence.item()

        action = self.label_map.get(
            class_index,
            "UNKNOWN",
        )

        return {
            "action": action,
            "class_index": class_index,
            "confidence": confidence,
        }