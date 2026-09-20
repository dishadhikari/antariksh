from pathlib import Path
import tempfile

from src.logging.logger import (
    ExperimentLogger,
)

from src.protocol.runtime import (
    ExperimentRuntime,
)


def main():

    sequence = [
        "START",
        "OPEN_BOX",
        "PICK_RED",
        "PICK_YELLOW",
        "COMPLETE",
    ]

    with tempfile.TemporaryDirectory() as temp_dir:

        database_path = (
            Path(temp_dir)
            / "experiment.db"
        )

        logger = ExperimentLogger(
            database_path=str(database_path)
        )

        runtime = ExperimentRuntime(
            experiment_id="box_experiment",
            sequence=sequence,
            logger=logger,
        )

        # Correct event
        result = runtime.process_event(
            "START",
            confidence=0.95,
        )

        assert result.status.value == "CORRECT"

        # Correct event
        result = runtime.process_event(
            "OPEN_BOX",
            confidence=0.91,
        )

        assert result.status.value == "CORRECT"

        # Wrong event
        result = runtime.process_event(
            "PICK_YELLOW",
            confidence=0.88,
        )

        assert (
            result.status.value
            == "OUT_OF_SEQUENCE"
        )

        # State must still expect PICK_RED.
        assert (
            runtime.expected_event
            == "PICK_RED"
        )

        events = logger.get_events(
            "box_experiment"
        )

        assert len(events) == 3

        assert (
            events[0]["detected_step"]
            == "START"
        )

        assert (
            events[2]["detected_step"]
            == "PICK_YELLOW"
        )

        assert (
            events[2]["status"]
            == "OUT_OF_SEQUENCE"
        )

        logger.close()

    print("Experiment runtime test passed.")


if __name__ == "__main__":
    main()