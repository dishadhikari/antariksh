import cv2
from pathlib import Path

from src.data.manifest import DatasetManifest


VIDEO_DIR = Path(
    "data/raw/box_experiment/videos"
)

MANIFEST_PATH = Path(
    "data/raw/box_experiment/metadata/manifest.json"
)


RECORDING_TYPES = {
    "1": "correct",
    "2": "correct",
    "3": "correct",
    "4": "correct",
    "5": "correct",

    "6": "skip_order",
    "7": "skip_order",
    "8": "skip_order",
    "9": "skip_order",
    "10": "skip_order",

    "11": "incomplete",
}

def get_video_info(video_path):
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    capture.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def main():

    manifest = DatasetManifest(
        MANIFEST_PATH
    )

    if not MANIFEST_PATH.exists():
        manifest.create(
            dataset_id="box_experiment_v1",
            description=(
                "BAS human activity recognition "
                "experiment dataset"
            ),
        )

    recordings = []

    for recording_number in range(1, 12):

        recording_id = (
            f"REC_{recording_number:06d}"
        )

        video_path = (
            VIDEO_DIR
            / f"{recording_number}.mp4"
        )

        if not video_path.exists():

            print(
                f"WARNING: Missing video: "
                f"{video_path}"
            )

            continue

        video_info = get_video_info(
            video_path
        )

        recording = {
            "recording_id": recording_id,

            "participant_id": "participant_01",

            "video": {
                "path": str(video_path),
                **video_info,
            },

            "execution": {
                "type": RECORDING_TYPES[
                    str(recording_number)
                ],
                "orientation": "normal",
            },

            "events": [],
        }

        recordings.append(recording)

    manifest.add_recordings(
        recordings
    )

    print()
    print(
        f"Registered {len(recordings)} recordings."
    )


if __name__ == "__main__":
    main()