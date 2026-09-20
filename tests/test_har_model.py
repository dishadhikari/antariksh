import torch

from src.har.model import TCN


def main():

    batch_size = 4
    sequence_length = 30
    feature_count = 8
    num_classes = 8

    model = TCN(
        input_features=feature_count,
        num_classes=num_classes,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        feature_count,
    )

    output = model(x)

    print("Input shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nModel:")
    print(model)

    assert output.shape == (
        batch_size,
        num_classes
    )


if __name__ == "__main__":
    main()