import cv2

from src.perception.tracker import ObjectTracker


def main():

    tracker = ObjectTracker(
        model_path="yolo11n.pt",
        confidence=0.5,
        device="cpu",
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam")

    print("ByteTrack test started.")
    print("Press Q to quit.")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                break

            tracked_objects = tracker.update(frame)

            output = tracker.draw(
                frame,
                tracked_objects,
            )

            cv2.imshow(
                "SIH - ByteTrack",
                output,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("ByteTrack test completed.")


if __name__ == "__main__":
    main()