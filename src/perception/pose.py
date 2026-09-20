from mmpose.apis import MMPoseInferencer


class PoseEstimator:

    def __init__(self, device="cpu"):
        self.device = device

        self.body_inferencer = MMPoseInferencer(
            pose2d="human",
            device=device,
            show_progress=False,
        )

        self.hand_inferencer = MMPoseInferencer(
            pose2d="hand",
            device=device,
            show_progress=False,
        )

    def predict_body(self, frame):
        if frame is None:
            raise ValueError("Frame cannot be None")

        results = self.body_inferencer(
            frame,
            return_vis=True,
        )

        return next(results)

    def predict_hands(self, frame):
        if frame is None:
            raise ValueError("Frame cannot be None")

        results = self.hand_inferencer(
            frame,
            return_vis=True,
        )

        return next(results)
    def predict(self, frame):
        return {
            "body": self.predict_body(frame),
            "hands": self.predict_hands(frame),
        }