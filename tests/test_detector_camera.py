import cv2

from src.camera.camera import Camera
from src.perception.detector import ObjectDetector


def main():
    camera = Camera(
        source=0,
        width=1280,
        height=720,
        fps=30,
    )

    detector = ObjectDetector(
        model_path="yolo11n.pt",
        confidence=0.5,
        device="cpu",
    )

    camera.open()

    print("YOLO camera test started.")
    print("Press Q to quit.")

    try:
        while True:
            frame = camera.read()

            if frame is None:
                print("Failed to read frame.")
                break

            result = detector.predict(frame)

            output = detector.draw(frame, result)

            cv2.imshow("SIH - YOLO Detection", output)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()