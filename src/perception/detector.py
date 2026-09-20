from ultralytics import YOLO


class ObjectDetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.5,
        device: str = "cpu",
    ):
        self.confidence = confidence
        self.device = device

        self.model = YOLO(model_path)

    def predict(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        return results[0]

    def draw(self, frame, result):
        return result.plot()