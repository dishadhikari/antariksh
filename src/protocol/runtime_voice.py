from src.protocol.runtime import ExperimentRuntime
from src.voice.alerts import ExperimentVoiceAlerts


class RuntimeVoiceController:

    def __init__(
        self,
        runtime: ExperimentRuntime,
        alerts: ExperimentVoiceAlerts,
    ):
        self.runtime = runtime
        self.alerts = alerts

    def process_event(
        self,
        event: str,
        confidence: float = 1.0,
    ):
        result = self.runtime.process_event(
            event=event,
            confidence=confidence,
        )

        if result.status.value == "CORRECT":
            self.alerts.correct(
                next_step=result.expected
            )

        elif result.status.value == "OUT_OF_SEQUENCE":
            self.alerts.wrong_sequence(
                expected_step=result.expected
            )

        elif result.status.value == "COMPLETED":
            self.alerts.completed()

        return result