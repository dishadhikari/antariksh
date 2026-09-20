import numpy as np

from src.har.dataset import HARDatasetBuilder


def main():

    sequence_length = 30

    # Create fake frame features.
    #
    # 100 frames
    # 8 features per frame
    features = np.random.rand(
        100,
        8
    ).astype(np.float32)

    # Every frame in this example
    # represents the same action.
    labels = [
        "PICK_RED"
        for _ in range(100)
    ]

    builder = HARDatasetBuilder(
        sequence_length=sequence_length,
        stride=5,
    )

    sequences, sequence_labels = (
        builder.create_sequences(
            features,
            labels
        )
    )

    print(
        "Sequences shape:",
        sequences.shape
    )

    print(
        "Labels:",
        sequence_labels
    )

    assert sequences.ndim == 3

    assert sequences.shape[1] == 30

    assert sequences.shape[2] == 8

    assert len(sequences) == len(
        sequence_labels
    )


if __name__ == "__main__":
    main()