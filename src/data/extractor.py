from pathlib import Path

import cv2


class FrameExtractor:

    def __init__(
        self,
        input_dir: str = "data/raw/box_experiment/videos",
        output_dir: str = "data/detection/images/raw",
        frame_interval: int = 5,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def extract_video(self, video_path: str | Path):
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video: {video_path}"
            )

        video_name = video_path.stem

        frame_index = 0
        saved_count = 0

        try:
            while True:

                success, frame = capture.read()

                if not success:
                    break

                if frame_index % self.frame_interval == 0:

                    output_path = (
                        self.output_dir
                        / f"{video_name}_{frame_index:06d}.jpg"
                    )

                    cv2.imwrite(
                        str(output_path),
                        frame
                    )

                    saved_count += 1

                frame_index += 1

        finally:
            capture.release()

        print(
            f"{video_name}: "
            f"{saved_count} frames extracted."
        )

        return saved_count

    def extract_all(self):
        videos = sorted(
            self.input_dir.glob("*.mp4")
        )

        if not videos:
            print(
                f"No videos found in {self.input_dir}"
            )
            return

        total = 0

        for video in videos:
            total += self.extract_video(video)

        print(
            f"Total frames extracted: {total}"
        )