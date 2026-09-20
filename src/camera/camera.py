import cv2


class Camera:
    def __init__(
        self,
        source=0,
        width=1280,
        height=720,
        fps=30,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps

        self.capture = None

    def open(self):
        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Unable to open camera source: {self.source}"
            )

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self):
        if self.capture is None:
            raise RuntimeError("Camera is not open")

        success, frame = self.capture.read()

        if not success:
            return None

        return frame

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None