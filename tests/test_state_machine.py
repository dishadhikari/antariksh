from src.protocol.state_machine import (
    ExperimentStateMachine,
    EventStatus,
)


def test_correct_sequence():

    sequence = [
        "START",
        "OPEN_BOX",
        "PICK_RED",
        "PICK_YELLOW",
    ]

    machine = ExperimentStateMachine(sequence)

    result = machine.process("START")
    assert result.status == EventStatus.CORRECT

    result = machine.process("OPEN_BOX")
    assert result.status == EventStatus.CORRECT

    result = machine.process("PICK_RED")
    assert result.status == EventStatus.CORRECT

    result = machine.process("PICK_YELLOW")
    assert result.status == EventStatus.COMPLETED


def test_wrong_sequence():

    sequence = [
        "START",
        "OPEN_BOX",
        "PICK_RED",
    ]

    machine = ExperimentStateMachine(sequence)

    machine.process("START")

    result = machine.process("PICK_RED")

    assert result.status == EventStatus.OUT_OF_SEQUENCE
    assert result.expected == "OPEN_BOX"
    assert result.detected == "PICK_RED"