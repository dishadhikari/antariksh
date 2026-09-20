import numpy as np

from src.har.train import HARTrainer


def main():

    np.random.seed(42)

    sequence_count = 100
    sequence_length = 30
    feature_count = 8
    num_classes = 4

    features = np.random.randn(
        sequence_count,
        sequence_length,
        feature_count,
    ).astype(np.float32)

    labels = [
        i % num_classes
        for i in range(sequence_count)
    ]

    train_features = features[:80]
    train_labels = labels[:80]

    val_features = features[80:]
    val_labels = labels[80:]

    trainer = HARTrainer(
        input_features=feature_count,
        num_classes=num_classes,
        learning_rate=0.001,
        batch_size=16,
        epochs=3,
        device="cpu",
    )

    trainer.fit(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        output_dir="models/har/test",
    )

    print()
    print("HAR training pipeline test passed.")


if __name__ == "__main__":
    main()