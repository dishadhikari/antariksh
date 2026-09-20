from src.logging.logger import ExperimentLogger
from src.protocol.runtime import ExperimentRuntime
from src.protocol.runtime_voice import RuntimeVoiceController


class MockAlerts:

    def __init__(self):
        self.messages = []

    def correct(self, next_step):
        self.messages.append(
            f"CORRECT:{next_step}"
        )

    def wrong_sequence(self, expected_step):
        self.messages.append(
            f"WRONG:{expected_step}"
        )

    def completed(self):
        self.messages.append(
            "COMPLETED"
        )


def main():

    logger = ExperimentLogger(
        "data/logs/test_runtime_voice.db"
    )

    runtime = ExperimentRuntime(
        experiment_id="test_experiment",
        sequence=[
            "START",
            "OPEN_BOX",
            "COMPLETE",
        ],
        logger=logger,
    )

    alerts = MockAlerts()

    controller = RuntimeVoiceController(
        runtime=runtime,
        alerts=alerts,
    )

    result = controller.process_event(
    "START",
    confidence=0.95,
)

    print("STATUS:", result.status.value)
    print("ALERTS:", alerts.messages)

    assert result.status.value == "CORRECT"
    assert alerts.messages[-1] == "CORRECT:OPEN_BOX"

    result = controller.process_event(
        "COMPLETE",
        confidence=0.90,
    )

    assert result.status.value == "OUT_OF_SEQUENCE"
    assert alerts.messages[-1] == "WRONG:OPEN_BOX"

    result = controller.process_event(
        "OPEN_BOX",
        confidence=0.95,
    )

    assert result.status.value == "CORRECT"

    result = controller.process_event(
        "COMPLETE",
        confidence=0.95,
    )

    assert result.status.value == "COMPLETED"
    assert alerts.messages[-1] == "COMPLETED"

    logger.close()

    print("Runtime + voice integration test passed.")


if __name__ == "__main__":
    main()