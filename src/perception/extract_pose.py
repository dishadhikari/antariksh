import json
from pathlib import Path

from mmpose.apis import MMPoseInferencer


INPUT_DIR = Path("data/detection/images/val")
OUTPUT_DIR = Path("data/pose/val")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inferencer = MMPoseInferencer(
        pose2d="rtmpose-m_8xb256-420e_body8-256x192"
    )

    images = sorted(INPUT_DIR.glob("*.jpg"))

    print(f"Found {len(images)} images")

    saved = 0

    for image_path in images:
        results = inferencer(str(image_path), show=False)

        for result in results:
            predictions = result["predictions"][0]

            if not predictions:
                print(f"No person: {image_path.name}")
                continue

            # Select the largest detected person
            primary = max(
                predictions,
                key=lambda p: (
                    p["bbox"][0][2] - p["bbox"][0][0]
                ) * (
                    p["bbox"][0][3] - p["bbox"][0][1]
                )
            )

            output = {
                "image": image_path.name,
                "keypoints": [
                    [float(x), float(y)]
                    for x, y in primary["keypoints"]
                ],
                "keypoint_scores": [
                    float(score)
                    for score in primary["keypoint_scores"]
                ],
                "bbox": [
                    [float(x) for x in box]
                    for box in primary["bbox"]
                ],
                "bbox_score": float(primary["bbox_score"]),
            }

            output_path = OUTPUT_DIR / f"{image_path.stem}.json"

            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

            saved += 1

    print(f"Saved {saved} pose files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()