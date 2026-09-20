import cv2

from config.loader import load_config
from camera.camera import Camera


def main():
    config = load_config("configs/experiment.yaml")

    camera_config = config["camera"]

    camera = Camera(
        source=camera_config["source"],
        width=camera_config["width"],
        height=camera_config["height"],
        fps=camera_config["fps"],
    )

    camera.open()

    print("Camera started.")
    print("Press Q to quit.")

    try:
        while True:
            frame = camera.read()

            if frame is None:
                print("Failed to read frame.")
                break

            cv2.imshow("SIH HAR - Camera", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()