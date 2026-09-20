from src.protocol.event_processor import (
    EventProcessor,
)
from src.protocol.state_machine import EventStatus


def main():

    sequence = [
        "START",
        "OPEN_BOX",
        "PICK_RED",
        "PICK_YELLOW",
        "PLACE_RED",
        "PLACE_YELLOW",
        "CLOSE_BOX",
        "COMPLETE",
    ]

    processor = EventProcessor(
        sequence
    )

    # First event
    result = processor.process(
        "START"
    )

    assert result.status == EventStatus.CORRECT
    assert result.detected == "START"

    print(result.message)

    # Correct second event
    result = processor.process(
        "OPEN_BOX"
    )

    assert result.status == EventStatus.CORRECT

    print(result.message)

    # Wrong event
    result = processor.process(
        "PLACE_RED"
    )

    assert result.status == EventStatus.OUT_OF_SEQUENCE

    print(result.message)

    # State should NOT advance after wrong event.
    assert processor.expected_event == "PICK_RED"

    # Continue correctly.
    result = processor.process(
        "PICK_RED"
    )

    assert result.status == EventStatus.CORRECT

    print(result.message)

    print()
    print("Event processor test passed.")


if __name__ == "__main__":
    main()