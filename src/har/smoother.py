from collections import deque, Counter


class PredictionSmoother:

    def __init__(
        self,
        window_size: int = 5,
        confidence_threshold: float = 0.60,
        min_votes: int = 3,
    ):
        if window_size <= 0:
            raise ValueError(
                "Window size must be positive"
            )

        if min_votes <= 0:
            raise ValueError(
                "Minimum votes must be positive"
            )

        if min_votes > window_size:
            raise ValueError(
                "min_votes cannot exceed window_size"
            )

        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.min_votes = min_votes

        self.predictions = deque(
            maxlen=window_size
        )

        self.last_emitted_action = None

    def update(
        self,
        action: str,
        confidence: float,
    ):

        if confidence < self.confidence_threshold:
            return {
                "stable": False,
                "action": None,
                "confidence": confidence,
                "event": None,
            }

        self.predictions.append(
            {
                "action": action,
                "confidence": confidence,
            }
        )

        if len(self.predictions) < self.min_votes:
            return {
                "stable": False,
                "action": None,
                "confidence": confidence,
                "event": None,
            }

        actions = [
            prediction["action"]
            for prediction in self.predictions
        ]

        counts = Counter(actions)

        stable_action, vote_count = (
            counts.most_common(1)[0]
        )

        if vote_count < self.min_votes:
            return {
                "stable": False,
                "action": None,
                "confidence": confidence,
                "event": None,
            }

        stable_predictions = [
            prediction["confidence"]
            for prediction in self.predictions
            if prediction["action"] == stable_action
        ]

        stable_confidence = (
            sum(stable_predictions)
            / len(stable_predictions)
        )

        event = None

        if stable_action != self.last_emitted_action:
            event = stable_action
            self.last_emitted_action = stable_action

        return {
            "stable": True,
            "action": stable_action,
            "confidence": stable_confidence,
            "event": event,
        }

    def reset(self):
        self.predictions.clear()
        self.last_emitted_action = None