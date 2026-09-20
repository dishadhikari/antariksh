from dataclasses import dataclass

from src.protocol.state_machine import (
    EventStatus,
    ExperimentStateMachine,
)


@dataclass
class EventProcessorResult:
    event: str
    status: EventStatus
    expected: str
    detected: str
    message: str


class EventProcessor:

    def __init__(
        self,
        sequence: list[str],
    ):
        self.state_machine = (
            ExperimentStateMachine(sequence)
        )

    def process(
        self,
        event: str,
    ) -> EventProcessorResult:

        result = self.state_machine.process(
            event
        )

        if result.status == EventStatus.CORRECT:

            message = (
                f"Correct step: {event}. "
                f"Next step: "
                f"{self.state_machine.expected_event}"
            )

        elif result.status == EventStatus.COMPLETED:

            message = (
                "Experiment completed successfully."
            )

        else:

            message = (
                f"Wrong sequence. "
                f"Expected: {result.expected}. "
                f"Detected: {result.detected}"
            )

        return EventProcessorResult(
        event=event,
        status=result.status,
        expected=(
            self.state_machine.expected_event
            if result.status == EventStatus.CORRECT
            else result.expected
        ),
        detected=result.detected,
        message=message,
    )

    @property
    def expected_event(self):
        return self.state_machine.expected_event

    @property
    def completed(self):
        return self.state_machine.completed

    def reset(self):
        self.state_machine.current_index = 0