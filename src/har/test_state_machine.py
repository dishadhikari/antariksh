from src.har.state_machine import ExperimentStateMachine


def test_sequence(name, actions):
    machine = ExperimentStateMachine()

    print(f"\n=== {name} ===")

    for action in actions:
        status = machine.update(action)

        print(
            f"{action:<22} -> "
            f"{status:<12} | "
            f"Next: {machine.expected_action()}"
        )

    print(f"Completed: {machine.completed}")
    print(f"Final status: {machine.status}")


def main():
    test_sequence(
        "CORRECT",
        [
            "LEFT_HAND_UP",
            "RIGHT_HAND_UP",
            "PICK_BLUE_BOX",
            "PUT_DOWN_BLUE_BOX",
            "PICK_BLACK_BOX",
            "PUT_DOWN_BLACK_BOX",
        ],
    )

    test_sequence(
        "SKIP_ORDER",
        [
            "LEFT_HAND_UP",
            "RIGHT_HAND_UP",
            "PUT_DOWN_BLUE_BOX",
        ],
    )

    test_sequence(
        "INCOMPLETE",
        [
            "LEFT_HAND_UP",
            "RIGHT_HAND_UP",
            "PICK_BLUE_BOX",
        ],
    )


if __name__ == "__main__":
    main()