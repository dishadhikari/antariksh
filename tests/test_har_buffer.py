import numpy as np

from src.har.buffer import TemporalFeatureBuffer


def main():

    buffer = TemporalFeatureBuffer(
        sequence_length=30,
        feature_count=8,
    )

    assert buffer.size == 0
    assert not buffer.is_ready

    # Add 29 frames.
    for _ in range(29):

        features = np.random.randn(
            8
        ).astype(np.float32)

        buffer.add(features)

    assert buffer.size == 29
    assert not buffer.is_ready

    # Add frame 30.
    buffer.add(
        np.random.randn(8).astype(np.float32)
    )

    assert buffer.size == 30
    assert buffer.is_ready

    sequence = buffer.get_sequence()

    print("Sequence shape:", sequence.shape)

    assert sequence.shape == (30, 8)

    # Add another frame.
    # Oldest frame should automatically disappear.
    buffer.add(
        np.random.randn(8).astype(np.float32)
    )

    assert buffer.size == 30

    sequence = buffer.get_sequence()

    assert sequence.shape == (30, 8)

    # Clear.
    buffer.clear()

    assert buffer.size == 0
    assert not buffer.is_ready

    print()
    print("Temporal buffer test passed.")


if __name__ == "__main__":
    main()