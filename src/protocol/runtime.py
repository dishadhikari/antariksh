from src.logging.logger import (
    ExperimentLogger,
)
from src.protocol.event_processor import (
    EventProcessor,
)


class ExperimentRuntime:

    def __init__(
        self,
        experiment_id: str,
        sequence: list[str],
        logger: ExperimentLogger,
    ):
        self.experiment_id = experiment_id

        self.processor = EventProcessor(
            sequence
        )

        self.logger = logger

    def process_event(
        self,
        event: str,
        confidence: float = 1.0,
    ):

        result = self.processor.process(
            event
        )

        self.logger.log_event(
            experiment_id=self.experiment_id,
            expected_step=result.expected,
            detected_step=result.detected,
            confidence=confidence,
            status=result.status.value,
            message=result.message,
        )

        return result

    @property
    def expected_event(self):
        return self.processor.expected_event

    @property
    def completed(self):
        return self.processor.completed

    def reset(self):
        self.processor.reset()