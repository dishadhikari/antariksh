import numpy as np

from src.har.features import HARFeatureExtractor
from src.har.buffer import TemporalFeatureBuffer
from src.har.inference import HARInference


class HARPipeline:

    def __init__(
        self,
        model_path: str,
        label_map: dict[int, str],
        sequence_length: int = 30,
        device: str = "cpu",
    ):
        self.feature_extractor = HARFeatureExtractor()

        self.buffer = TemporalFeatureBuffer(
            sequence_length=sequence_length,
            feature_count=self.feature_extractor.feature_count,
        )

        self.inference = HARInference(
            model_path=model_path,
            label_map=label_map,
            device=device,
        )

    def process(self, interaction_features):

        # Convert InteractionFeatures → 8-value vector
        feature_vector = (
            self.feature_extractor.extract(
                interaction_features
            )
        )

        # Add current frame
        self.buffer.add(feature_vector)

        # Not enough temporal information yet
        if not self.buffer.is_ready:

            return {
                "ready": False,
                "action": None,
                "confidence": 0.0,
                "buffer_size": self.buffer.size,
            }

        # Get [30, 8]
        sequence = self.buffer.get_sequence()

        # TCN inference
        prediction = self.inference.predict(
            sequence
        )

        return {
            "ready": True,
            "action": prediction["action"],
            "confidence": prediction["confidence"],
            "class_index": prediction["class_index"],
            "buffer_size": self.buffer.size,
        }

    def reset(self):
        self.buffer.clear()