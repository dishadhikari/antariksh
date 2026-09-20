import cv2

from src.camera.camera import Camera
from src.perception.pose import PoseEstimator
from src.interaction.adapter import PerceptionAdapter


def main():

    camera = Camera(
        source=0,
        width=640,
        height=480,
        fps=30,
    )

    pose = PoseEstimator(device="cpu")
    adapter = PerceptionAdapter()

    camera.open()

    print("RTMPose test started.")

    frame_count = 0

    try:

        while True:

            frame = camera.read()

            if frame is None:
                print("Failed to read frame.")
                break

            frame_count += 1

            if frame_count % 3 == 0:

                result = pose.predict(frame)

                hands = adapter.hands_from_pose(result)

                print("Detected hands:")

                for hand in hands:
                    print(
                        f"{hand.name}: "
                        f"position={hand.position}, "
                        f"confidence={hand.confidence:.3f}"
                    )

                # Close immediately after successful test
                break

            cv2.imshow("SIH - RTMPose", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()

        print("RTMPose test finished.")


if __name__ == "__main__":
    main()