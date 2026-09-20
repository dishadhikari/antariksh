from src.protocol.config import ExperimentConfig


def main():

    config = ExperimentConfig(
        "experiments/box_experiment/experiment.yaml"
    )

    print("Experiment:")
    print(config.experiment_id)

    print()
    print("Name:")
    print(config.experiment_name)

    print()
    print("Sequence:")

    for index, step in enumerate(
        config.sequence,
        start=1,
    ):
        print(
            f"{index}. {step}"
        )

    print()
    print(
        "Confidence threshold:",
        config.confidence_threshold,
    )

    print(
        "Temporal window:",
        config.temporal_window,
    )

    print(
        "Minimum votes:",
        config.minimum_votes,
    )

    assert config.experiment_id == (
        "box_experiment"
    )

    assert len(config.sequence) == 8

    assert config.confidence_threshold == 0.60
    assert config.temporal_window == 5
    assert config.minimum_votes == 3

    print()
    print("Experiment config test passed.")


if __name__ == "__main__":
    main()