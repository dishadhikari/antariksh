from src.camera.camera import Camera
from src.perception.pipeline import PerceptionPipeline
from src.har.features import HARFeatureExtractor
from src.har.buffer import TemporalFeatureBuffer


def main():

    camera = Camera(
        source=0,
        width=640,
        height=480,
        fps=15,
    )

    perception = PerceptionPipeline(
        detector_model="yolo11n.pt",
        confidence=0.5,
        device="cpu",
    )

    feature_extractor = HARFeatureExtractor()

    buffer = TemporalFeatureBuffer(
        sequence_length=30,
        feature_count=feature_extractor.feature_count,
    )

    camera.open()

    print("Real-time HAR buffer test started.")
    print("Press Q to quit.")

    frame_count = 0

    try:

        while True:

            frame = camera.read()

            if frame is None:
                print("Failed to read frame.")
                break

            frame_count += 1

            # CPU optimization:
            # Run expensive perception every 3rd frame.
            if frame_count % 3 != 0:
                continue

            result = perception.process(frame)

            feature_vector = feature_extractor.extract(
                result.interaction_features
            )

            buffer.add(feature_vector)

            print(
                f"Frame: {frame_count} | "
                f"Buffer: {buffer.size}/30 | "
                f"Ready: {buffer.is_ready}"
            )

            if buffer.is_ready:

                sequence = buffer.get_sequence()

                print(
                    "HAR sequence ready:",
                    sequence.shape
                )

                break

    finally:

        camera.release()

        print()
        print("Real-time HAR buffer test finished.")


if __name__ == "__main__":
    main()