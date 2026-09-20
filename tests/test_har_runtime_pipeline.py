import torch

from src.har.model import TCN
from src.har.runtime_pipeline import HARRuntimePipeline
from src.logging.logger import ExperimentLogger


MODEL_PATH = "models/har/test_runtime_tcn.pt"


def create_test_model():

    model = TCN(
        input_features=8,
        num_classes=3,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_features": 8,
            "num_classes": 3,
        },
        MODEL_PATH,
    )


def main():

    create_test_model()

    label_map = {
        0: "START",
        1: "OPEN_BOX",
        2: "COMPLETE",
    }

    logger = ExperimentLogger(
        "data/logs/test_har_runtime.db"
    )

    pipeline = HARRuntimePipeline(
        model_path=MODEL_PATH,
        label_map=label_map,
        experiment_id="test_experiment",
        sequence=[
            "START",
            "OPEN_BOX",
            "COMPLETE",
        ],
        logger=logger,
        sequence_length=30,
        confidence_threshold=0.0,
        smoothing_window=3,
        minimum_votes=2,
        device="cpu",
    )

    from src.interaction.interaction_engine import InteractionFeatures

    features = InteractionFeatures(
        left_hand_red_distance=100,
        right_hand_red_distance=100,
        left_hand_yellow_distance=100,
        right_hand_yellow_distance=100,
        left_hand_experiment_box_distance=100,
        right_hand_experiment_box_distance=100,
        left_hand_speed=10,
        right_hand_speed=10,
    )

    print("Feeding feature frames...")

    for index in range(35):

        result = pipeline.process(features)

        if result["event"] is not None:
            print(
                f"Event detected: {result['event']}"
            )

            print(
                f"Validation: "
                f"{result['validation'].status.value}"
            )

    logger.close()

    print()
    print("HAR runtime pipeline test passed.")


if __name__ == "__main__":
    main()