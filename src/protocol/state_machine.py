from dataclasses import dataclass
from enum import Enum


class EventStatus(Enum):
    CORRECT = "CORRECT"
    OUT_OF_SEQUENCE = "OUT_OF_SEQUENCE"
    COMPLETED = "COMPLETED"


@dataclass
class ValidationResult:
    status: EventStatus
    expected: str
    detected: str


class ExperimentStateMachine:

    def __init__(self, sequence: list[str]):
        if not sequence:
            raise ValueError("Experiment sequence cannot be empty")

        self.sequence = sequence
        self.current_index = 0

    @property
    def expected_event(self) -> str:
        return self.sequence[self.current_index]

    @property
    def completed(self) -> bool:
        return self.current_index >= len(self.sequence) - 1

    def process(self, detected_event: str) -> ValidationResult:

        expected = self.expected_event

        if detected_event == expected:

            if self.current_index == len(self.sequence) - 1:
                return ValidationResult(
                    status=EventStatus.COMPLETED,
                    expected=expected,
                    detected=detected_event,
                )

            self.current_index += 1

            return ValidationResult(
                status=EventStatus.CORRECT,
                expected=expected,
                detected=detected_event,
            )

        return ValidationResult(
            status=EventStatus.OUT_OF_SEQUENCE,
            expected=expected,
            detected=detected_event,
        )