from src.har.pipeline import HARPipeline
from src.har.smoother import PredictionSmoother
from src.protocol.runtime import ExperimentRuntime


class HARRuntimePipeline:

    def __init__(
        self,
        model_path,
        label_map,
        experiment_id,
        sequence,
        logger,
        sequence_length=30,
        confidence_threshold=0.60,
        smoothing_window=5,
        minimum_votes=3,
        device="cpu",
    ):

        self.har = HARPipeline(
            model_path=model_path,
            label_map=label_map,
            sequence_length=sequence_length,
            device=device,
        )

        self.smoother = PredictionSmoother(
            window_size=smoothing_window,
            confidence_threshold=confidence_threshold,
            min_votes=minimum_votes,
        )

        self.runtime = ExperimentRuntime(
            experiment_id=experiment_id,
            sequence=sequence,
            logger=logger,
        )

    def process(self, interaction_features):

        prediction = self.har.process(
            interaction_features
        )

        if not prediction["ready"]:
            return {
                "ready": False,
                "event": None,
                "prediction": prediction,
                "validation": None,
            }

        smoothed = self.smoother.update(
            action=prediction["action"],
            confidence=prediction["confidence"],
        )

        if smoothed["event"] is None:
            return {
                "ready": True,
                "event": None,
                "prediction": prediction,
                "smoothing": smoothed,
                "validation": None,
            }

        validation = self.runtime.process_event(
            event=smoothed["event"],
            confidence=smoothed["confidence"],
        )

        return {
            "ready": True,
            "event": smoothed["event"],
            "prediction": prediction,
            "smoothing": smoothed,
            "validation": validation,
        }

    def reset(self):
        self.har.reset()
        self.smoother.reset()
        self.runtime.reset()