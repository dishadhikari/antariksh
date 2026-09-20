from src.har.runtime_pipeline import HARRuntimePipeline
from src.voice.alerts import ExperimentVoiceAlerts


class VoiceRuntimePipeline:

    def __init__(
        self,
        har_runtime: HARRuntimePipeline,
        alerts: ExperimentVoiceAlerts,
    ):
        self.har_runtime = har_runtime
        self.alerts = alerts

    def process(self, interaction_features):

        result = self.har_runtime.process(
            interaction_features
        )

        validation = result.get("validation")

        if validation is None:
            return result

        status = validation.status.value

        if status == "CORRECT":
            self.alerts.correct(
                validation.expected
            )

        elif status == "OUT_OF_SEQUENCE":
            self.alerts.wrong_sequence(
                validation.expected
            )

        elif status == "COMPLETED":
            self.alerts.completed()

        return result

    def reset(self):
        self.har_runtime.reset()