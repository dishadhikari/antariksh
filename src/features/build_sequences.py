import json
from collections import defaultdict
from pathlib import Path

INPUT_DIR = Path("data/features/val")
OUTPUT_DIR = Path("data/sequences/val")

WINDOW_SIZE = 5


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.json"))
    print(f"Feature files: {len(files)}")

    # Group frames by recording ID.
    recordings = defaultdict(list)

    for file in files:
        # Example: 9_000123.json -> recording "9"
        recording_id = file.stem.split("_")[0]
        recordings[recording_id].append(file)

    sequences = []

    for recording_id, recording_files in sorted(recordings.items()):
        recording_files.sort()

        if len(recording_files) < WINDOW_SIZE:
            continue

        for i in range(len(recording_files) - WINDOW_SIZE + 1):
            window_files = recording_files[i:i + WINDOW_SIZE]

            frames = []

            for file in window_files:
                with open(file, "r") as f:
                    frames.append(json.load(f))

            sequences.append({
                "recording_id": recording_id,
                "start_frame": frames[0]["image"],
                "end_frame": frames[-1]["image"],
                "frames": frames,
            })

    output_file = OUTPUT_DIR / "sequences.json"

    with open(output_file, "w") as f:
        json.dump(sequences, f, indent=2)

    print(f"Recordings: {len(recordings)}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Sequences: {len(sequences)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()