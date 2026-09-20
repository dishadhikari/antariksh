from dataclasses import dataclass

from ultralytics import YOLO


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


class ObjectTracker:

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.5,
        device: str = "cpu",
        tracker: str = "bytetrack.yaml",
    ):
        self.confidence = confidence
        self.device = device
        self.tracker = tracker

        self.model = YOLO(model_path)

    def update(self, frame) -> list[TrackedObject]:

        results = self.model.track(
            source=frame,
            conf=self.confidence,
            device=self.device,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None:
            return []

        boxes = result.boxes

        if boxes.id is None:
            return []

        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()

        names = result.names

        tracked_objects = []

        for track_id, class_id, confidence, bbox in zip(
            track_ids,
            class_ids,
            confidences,
            coordinates,
        ):
            tracked_objects.append(
                TrackedObject(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=names[class_id],
                    confidence=float(confidence),
                    bbox=tuple(
                        float(value)
                        for value in bbox
                    ),
                )
            )

        return tracked_objects

    def draw(self, frame, tracked_objects):

        for obj in tracked_objects:

            x1, y1, x2, y2 = map(
                int,
                obj.bbox,
            )

            label = (
                f"{obj.class_name} "
                f"ID:{obj.track_id} "
                f"{obj.confidence:.2f}"
            )

            import cv2

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return frame