EXPECTED_SEQUENCE = [
    "LEFT_HAND_UP",
    "RIGHT_HAND_UP",
    "PICK_BLUE_BOX",
    "PUT_DOWN_BLUE_BOX",
    "PICK_BLACK_BOX",
    "PUT_DOWN_BLACK_BOX",
]


class ExperimentStateMachine:
    def __init__(self):
        self.current_index = 0
        self.completed = []
        self.status = "RUNNING"

    def update(self, action):
        if self.status != "RUNNING":
            return self.status

        expected = EXPECTED_SEQUENCE[self.current_index]

        if action == expected:
            self.completed.append(action)
            self.current_index += 1

            if self.current_index == len(EXPECTED_SEQUENCE):
                self.status = "COMPLETE"

        elif action in EXPECTED_SEQUENCE:
            expected_index = EXPECTED_SEQUENCE.index(action)

            if expected_index > self.current_index:
                self.status = "SKIP_ORDER"

        return self.status

    def expected_action(self):
        if self.current_index >= len(EXPECTED_SEQUENCE):
            return None

        return EXPECTED_SEQUENCE[self.current_index]