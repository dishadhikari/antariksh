import torch

from src.har.model import TCN
from src.har.inference import HARInference
from src.har.buffer import TemporalFeatureBuffer


MODEL_PATH = "models/har/test_tcn.pt"


def create_test_model():

    model = TCN(
        input_features=8,
        num_classes=8,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_features": 8,
        "num_classes": 8,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )


def main():

    print("Creating temporary TCN checkpoint...")

    create_test_model()

    label_map = {
        0: "START",
        1: "OPEN_BOX",
        2: "PICK_RED",
        3: "PICK_YELLOW",
        4: "PLACE_RED",
        5: "PLACE_YELLOW",
        6: "CLOSE_BOX",
        7: "COMPLETE",
    }

    inference = HARInference(
        model_path=MODEL_PATH,
        label_map=label_map,
        device="cpu",
    )

    buffer = TemporalFeatureBuffer(
        sequence_length=30,
        feature_count=8,
    )

    print("Adding feature frames...")

    for _ in range(30):

        features = torch.randn(8).numpy()

        buffer.add(features)

    assert buffer.is_ready

    sequence = buffer.get_sequence()

    print("Sequence shape:", sequence.shape)

    prediction = inference.predict(sequence)

    print()
    print("TCN prediction:")
    print("Action:", prediction["action"])
    print("Class:", prediction["class_index"])
    print("Confidence:", prediction["confidence"])

    print()
    print("Real-time TCN inference test passed.")


if __name__ == "__main__":
    main()