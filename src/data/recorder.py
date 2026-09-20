from datetime import datetime
from pathlib import Path

import cv2

from src.camera.camera import Camera
from src.data.manifest import DatasetManifest


class DatasetRecorder:

    def __init__(
        self,
        output_dir: str = "data/raw/box_experiment",
        camera_source=0,
        width=1280,
        height=720,
        fps=30,
    ):
        self.output_dir = Path(output_dir)

        self.video_dir = self.output_dir / "videos"
        self.metadata_dir = self.output_dir / "metadata"

        self.video_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.metadata_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.camera = Camera(
            source=camera_source,
            width=width,
            height=height,
            fps=fps
        )

        self.fps = fps
        self.width = width
        self.height = height

    def record(
        self,
        recording_id: str,
        participant_id: str,
        execution_type: str,
        orientation: str = "normal",
    ):
        video_path = (
            self.video_dir /
            f"{recording_id}.mp4"
        )

        manifest_path = (
            self.metadata_dir /
            "manifest.json"
        )

        manifest = DatasetManifest(manifest_path)

        if not manifest_path.exists():
            manifest.create(
                dataset_id="box_experiment_v1",
                description="Custom BAS experiment dataset"
            )

        self.camera.open()

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height)
        )

        if not writer.isOpened():
            self.camera.release()
            raise RuntimeError(
                f"Unable to create video file: {video_path}"
            )

        print("Recording started.")
        print("Perform the experiment.")
        print("Press Q to stop recording.")

        frame_count = 0

        try:
            while True:

                frame = self.camera.read()

                if frame is None:
                    print("Failed to read frame.")
                    break

                display_frame = frame.copy()

                cv2.putText(
                    display_frame,
                    f"REC {recording_id}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    display_frame,
                    "Press Q to stop",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                writer.write(frame)

                cv2.imshow(
                    "SIH Dataset Recorder",
                    display_frame
                )

                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            writer.release()
            self.camera.release()
            cv2.destroyAllWindows()

        recording = {
            "recording_id": recording_id,
            "participant_id": participant_id,

            "video": {
                "path": str(video_path),
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "frame_count": frame_count
            },

            "execution": {
                "type": execution_type,
                "orientation": orientation
            },

            "created_at": datetime.now().isoformat(),

            "events": []
        }

        manifest.add_recording(recording)

        print()
        print(f"Saved recording: {video_path}")
        print(f"Frames recorded: {frame_count}")