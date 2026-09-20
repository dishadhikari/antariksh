from src.data.recorder import DatasetRecorder


def main():

    recorder = DatasetRecorder(
        output_dir="data/raw/box_experiment",
        camera_source=0,
        width=1280,
        height=720,
        fps=30,
    )

    recorder.record(
        recording_id="REC_000001",
        participant_id="P001",
        execution_type="correct",
        orientation="normal",
    )


if __name__ == "__main__":
    main()