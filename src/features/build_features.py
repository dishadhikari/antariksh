import json
from pathlib import Path

import sys

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"

POSE_DIR = Path(f"data/pose/{SPLIT}")
YOLO_DIR = Path("../runs/detect/runs/detection_features/v2/labels")
OUTPUT_DIR = Path(f"data/features/{SPLIT}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pose_files = sorted(POSE_DIR.glob("*.json"))

    print(f"Pose files: {len(pose_files)}")

    for pose_file in pose_files:
        with open(pose_file, "r") as f:
            pose = json.load(f)

        image_name = pose["image"]
        stem = Path(image_name).stem

        yolo_file = YOLO_DIR / f"{stem}.txt"

        blue_box = [0.0, 0.0, 0.0, 0.0, 0.0]
        black_box = [0.0, 0.0, 0.0, 0.0, 0.0]

        if yolo_file.exists():
            for line in yolo_file.read_text().splitlines():
                parts = line.split()

                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                values = [float(x) for x in parts[1:5]]

                if class_id == 0:
                    blue_box = [1.0] + values

                elif class_id == 1 and black_box[0] == 0.0:
                    black_box = [1.0] + values

        # Person bounding box
        bbox = pose["bbox"][0]
        x1, y1, x2, y2 = bbox

        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)

        # Normalize keypoints relative to the person's bounding box
        keypoints = []

        for x, y in pose["keypoints"]:
            nx = (float(x) - x1) / width
            ny = (float(y) - y1) / height

            keypoints.extend([nx, ny])

        # Keypoint confidence scores
        scores = [
            float(score)
            for score in pose["keypoint_scores"]
        ]

        feature = {
            "image": image_name,
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "blue_box": blue_box,
            "black_box": black_box,
        }

        output_file = OUTPUT_DIR / f"{stem}.json"

        with open(output_file, "w") as f:
            json.dump(feature, f, indent=2)

    print(f"Features saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()