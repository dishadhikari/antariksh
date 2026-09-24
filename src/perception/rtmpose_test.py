from pathlib import Path

from mmpose.apis import MMPoseInferencer


def main():
    output_dir = Path("runs/pose")
    output_dir.mkdir(parents=True, exist_ok=True)

    inferencer = MMPoseInferencer(
        pose2d="rtmpose-m_8xb256-420e_body8-256x192"
    )

    image = "data/detection/images/test/7_000185.jpg"

    results = inferencer(
        image,
        out_dir=str(output_dir),
        vis_out_dir=str(output_dir / "visualized"),
    )

    for result in results:
        print("RTMPose result generated.")
        print(result)


if __name__ == "__main__":
    main()