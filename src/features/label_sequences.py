import json
from pathlib import Path

import yaml

ANNOTATIONS = Path("data/annotations/correct_actions.yaml")

SETS = {
    "train": Path("data/sequences/train/sequences.json"),
    "val": Path("data/sequences/val/sequences.json"),
}

OUTPUT_DIR = Path("data/sequences/labeled")


def get_recording_and_frame(image_name):
    # Example: 1_000185.jpg
    stem = Path(image_name).stem
    recording, frame = stem.split("_", 1)
    return recording, int(frame)


def find_action(recording, frame, annotations):
    for action in annotations["recordings"].get(recording, []):
        if action["start"] <= frame <= action["end"]:
            return action["name"]

    return None


def label_set(set_name, input_file):
    with open(input_file, "r") as f:
        sequences = json.load(f)

    labeled = []

    for sequence in sequences:
        start_recording, start_frame = get_recording_and_frame(
            sequence["start_frame"]
        )
        end_recording, end_frame = get_recording_and_frame(
            sequence["end_frame"]
        )

        # Never label a sequence crossing recordings.
        if start_recording != end_recording:
            continue

        center_frame = (start_frame + end_frame) // 2

        action = find_action(
            start_recording,
            center_frame,
            annotations,
        )

        if action is None:
            continue

        sequence["label"] = action
        sequence["recording"] = start_recording
        sequence["center_frame"] = center_frame

        labeled.append(sequence)

    output_file = OUTPUT_DIR / f"{set_name}_labeled.json"

    with open(output_file, "w") as f:
        json.dump(labeled, f, indent=2)

    print(f"{set_name}:")
    print(f"  Input sequences: {len(sequences)}")
    print(f"  Labeled sequences: {len(labeled)}")
    print(f"  Saved to: {output_file}")

    counts = {}

    for sequence in labeled:
        label = sequence["label"]
        counts[label] = counts.get(label, 0) + 1

    print("  Labels:")

    for label, count in counts.items():
        print(f"    {label}: {count}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global annotations

    with open(ANNOTATIONS, "r") as f:
        annotations = yaml.safe_load(f)

    for set_name, input_file in SETS.items():
        label_set(set_name, input_file)


if __name__ == "__main__":
    main()