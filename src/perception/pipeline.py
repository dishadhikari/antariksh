from dataclasses import dataclass

from src.perception.detector import ObjectDetector
from src.perception.tracker import ObjectTracker
from src.perception.pose import PoseEstimator
from src.interaction.adapter import PerceptionAdapter
from src.interaction.interaction_engine import (
    InteractionEngine,
    InteractionFeatures,
)


@dataclass
class PerceptionResult:
    detections: object
    tracked_objects: list
    pose_result: object
    interaction_features: InteractionFeatures


class PerceptionPipeline:

    def __init__(
        self,
        detector_model="yolo11n.pt",
        confidence=0.5,
        device="cpu",
    ):
        self.detector = ObjectDetector(
            model_path=detector_model,
            confidence=confidence,
            device=device,
        )

        self.tracker = ObjectTracker(
            model_path=detector_model,
            confidence=confidence,
            device=device,
        )

        self.pose = PoseEstimator(
            device=device,
        )

        self.adapter = PerceptionAdapter()
        self.interaction = InteractionEngine()

    def process(self, frame):

        if frame is None:
            raise ValueError("Frame cannot be None")

        detection_result = self.detector.predict(frame)

        tracked_objects = self.tracker.update(frame)

        pose_result = self.pose.predict(frame)

        objects = self.adapter.objects_from_tracking(
            tracked_objects
        )

        hands = self.adapter.hands_from_pose(
            pose_result
        )

        interaction_features = self.interaction.extract(
            hands=hands,
            objects=objects,
        )

        return PerceptionResult(
            detections=detection_result,
            tracked_objects=tracked_objects,
            pose_result=pose_result,
            interaction_features=interaction_features,
        )