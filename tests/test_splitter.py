from src.data.splitter import DatasetSplitter


def main():

    splitter = DatasetSplitter(
        manifest_path=(
            "data/raw/box_experiment/"
            "metadata/manifest.json"
        ),
        output_path="data/splits.json",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    result = splitter.split()

    print("\nDataset split complete.")

    print(
        f"Train recordings: "
        f"{len(result['recordings']['train'])}"
    )

    print(
        f"Validation recordings: "
        f"{len(result['recordings']['val'])}"
    )

    print(
        f"Test recordings: "
        f"{len(result['recordings']['test'])}"
    )

    print(
        f"\nSaved to: data/splits.json"
    )


if __name__ == "__main__":
    main()