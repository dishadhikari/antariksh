from src.protocol.voice_runtime import VoiceRuntimePipeline


class MockAlerts:

    def __init__(self):
        self.calls = []

    def correct(self, next_step):
        self.calls.append(
            ("CORRECT", next_step)
        )

    def wrong_sequence(self, expected_step):
        self.calls.append(
            ("WRONG", expected_step)
        )

    def completed(self):
        self.calls.append(
            ("COMPLETED",)
        )


class MockHARRuntime:

    def __init__(self):
        self.result = None

    def process(self, features):
        return self.result

    def reset(self):
        pass


class MockStatus:

    def __init__(self, value):
        self.value = value


class MockValidation:

    def __init__(
        self,
        status,
        expected,
    ):
        self.status = MockStatus(status)
        self.expected = expected


def main():

    har_runtime = MockHARRuntime()
    alerts = MockAlerts()

    pipeline = VoiceRuntimePipeline(
        har_runtime=har_runtime,
        alerts=alerts,
    )

    # Correct
    har_runtime.result = {
        "validation": MockValidation(
            "CORRECT",
            "OPEN_BOX",
        )
    }

    pipeline.process(None)

    assert alerts.calls[-1] == (
        "CORRECT",
        "OPEN_BOX",
    )

    # Wrong sequence
    har_runtime.result = {
        "validation": MockValidation(
            "OUT_OF_SEQUENCE",
            "OPEN_BOX",
        )
    }

    pipeline.process(None)

    assert alerts.calls[-1] == (
        "WRONG",
        "OPEN_BOX",
    )

    # Completed
    har_runtime.result = {
        "validation": MockValidation(
            "COMPLETED",
            "COMPLETE",
        )
    }

    pipeline.process(None)

    assert alerts.calls[-1] == (
        "COMPLETED",
    )

    print("Voice runtime pipeline test passed.")


if __name__ == "__main__":
    main()