from src.interaction.interaction_engine import (
    ObjectInfo,
    HandInfo,
)


class PerceptionAdapter:

    def objects_from_tracking(self, tracked_objects):
        objects = []

        for tracked in tracked_objects:
            objects.append(
                ObjectInfo(
                    name=tracked.class_name,
                    bbox=tracked.bbox,
                    confidence=tracked.confidence,
                    track_id=tracked.track_id,
                )
            )

        return objects

    def hands_from_pose(self, pose_result):
        hands = []

        if not pose_result:
            return hands

        hands_result = pose_result.get("hands")

        if not hands_result:
            return hands

        predictions = hands_result.get("predictions", [])

        if not predictions:
            return hands

        hand_predictions = predictions[0]

        for index, prediction in enumerate(hand_predictions):

            keypoints = prediction.get("keypoints", [])
            scores = prediction.get("keypoint_scores", [])

            if not keypoints or not scores:
                continue

            # Use the mean keypoint position as the hand position.
            x_values = [point[0] for point in keypoints]
            y_values = [point[1] for point in keypoints]

            x = sum(x_values) / len(x_values)
            y = sum(y_values) / len(y_values)

            confidence = sum(scores) / len(scores)

            if confidence < 0.30:
                continue

            # Temporary left/right assignment.
            # Image-space left = smaller x coordinate.
            name = "left" if index == 0 else "right"

            hands.append(
                HandInfo(
                    name=name,
                    position=(float(x), float(y)),
                    confidence=float(confidence),
                )
            )

        return hands