import cv2

from src.camera.camera import Camera
from src.perception.pipeline import PerceptionPipeline


def main():

    camera = Camera(
        source=0,
        width=640,
        height=480,
        fps=15,
    )

    pipeline = PerceptionPipeline(
        detector_model="yolo11n.pt",
        confidence=0.5,
        device="cpu",
    )

    camera.open()

    print("Perception pipeline test started.")

    try:

        frame = camera.read()

        if frame is None:
            raise RuntimeError("Failed to read camera frame.")

        result = pipeline.process(frame)

        features = result.interaction_features

        print()
        print("Tracked objects:", len(result.tracked_objects))
        print("Interaction features:")
        print(features)
        print()
        print("Perception pipeline test passed.")

    finally:

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()