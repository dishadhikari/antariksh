from pathlib import Path
import tempfile

from src.logging.logger import (
    ExperimentLogger,
)


def main():

    with tempfile.TemporaryDirectory() as temp_dir:

        database_path = (
            Path(temp_dir)
            / "experiment.db"
        )

        json_path = (
            Path(temp_dir)
            / "events.json"
        )

        logger = ExperimentLogger(
            database_path=str(database_path)
        )

        logger.log_event(
            experiment_id="box_experiment",
            expected_step="OPEN_BOX",
            detected_step="OPEN_BOX",
            confidence=0.94,
            status="CORRECT",
            message="Correct step.",
        )

        logger.log_event(
            experiment_id="box_experiment",
            expected_step="PICK_RED",
            detected_step="PICK_YELLOW",
            confidence=0.87,
            status="OUT_OF_SEQUENCE",
            message=(
                "Expected PICK_RED, "
                "detected PICK_YELLOW."
            ),
        )

        events = logger.get_events(
            "box_experiment"
        )

        assert len(events) == 2

        assert (
            events[0]["status"]
            == "CORRECT"
        )

        assert (
            events[1]["status"]
            == "OUT_OF_SEQUENCE"
        )

        assert (
            events[1]["confidence"]
            == 0.87
        )

        logger.export_json(
            str(json_path),
            "box_experiment",
        )

        assert json_path.exists()

        logger.close()

        assert database_path.exists()

    print("Experiment logger test passed.")


if __name__ == "__main__":
    main()