import cv2
from pathlib import Path


CLASS_NAMES = {
    0: "blue_box",
    1: "black_box",
}


IMAGE_ROOT = Path("data/detection/images")


class YOLOAnnotator:

    def __init__(self):

        self.image_paths = []

        for split in ["train", "val", "test"]:
            self.image_paths.extend(
                sorted(
                    (IMAGE_ROOT / split).glob("*.jpg")
                )
            )

        if not self.image_paths:
            raise RuntimeError("No images found.")

        self.index = 0

        self.image = None
        self.display = None

        self.boxes = []

        self.drawing = False
        self.start_x = 0
        self.start_y = 0

        self.current_class = 0

    def get_split(self, image_path):

        for split in ["train", "val", "test"]:
            if (
                IMAGE_ROOT
                / split
                / image_path.name
            ).exists():
                return split

        raise RuntimeError(
            f"Unable to determine split: {image_path}"
        )

    def load_image(self):

        image_path = self.image_paths[self.index]

        self.image = cv2.imread(str(image_path))

        if self.image is None:
            raise RuntimeError(
                f"Unable to read: {image_path}"
            )

        self.display = self.image.copy()

        self.load_existing_labels()

    def label_path(self):

        image_path = self.image_paths[self.index]

        split = self.get_split(image_path)

        label_dir = Path("data/detection/labels") / split

        label_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        return label_dir / f"{image_path.stem}.txt"

    def load_existing_labels(self):

        self.boxes = []

        path = self.label_path()

        if not path.exists():
            return

        with path.open("r", encoding="utf-8") as file:

            for line in file:

                values = line.strip().split()

                if len(values) != 5:
                    continue

                class_id = int(values[0])

                if class_id not in CLASS_NAMES:
                    continue

                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])

                self.boxes.append(
                    (
                        class_id,
                        x_center,
                        y_center,
                        width,
                        height,
                    )
                )

    def save_labels(self):

        path = self.label_path()

        height, width = self.image.shape[:2]

        with path.open("w", encoding="utf-8") as file:

            for (
                class_id,
                x_center,
                y_center,
                box_width,
                box_height,
            ) in self.boxes:

                file.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}\n"
                )

    def draw(self):

        self.display = self.image.copy()

        for box in self.boxes:

            (
                class_id,
                x_center,
                y_center,
                width,
                height,
            ) = box

            image_height, image_width = self.image.shape[:2]

            x1 = int(
                (x_center - width / 2)
                * image_width
            )

            y1 = int(
                (y_center - height / 2)
                * image_height
            )

            x2 = int(
                (x_center + width / 2)
                * image_width
            )

            y2 = int(
                (y_center + height / 2)
                * image_height
            )

            cv2.rectangle(
                self.display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                self.display,
                CLASS_NAMES[class_id],
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        image_name = self.image_paths[self.index].name
        split = self.get_split(self.image_paths[self.index])

        instructions = (
            "1=BLUE  2=BLACK | "
            "S=SAVE/NEXT | "
            "D=DELETE LAST | "
            "Q=QUIT"
        )

        cv2.putText(
            self.display,
            instructions,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            self.display,
            f"{split.upper()} | "
            f"{self.index + 1}/"
            f"{len(self.image_paths)} | "
            f"{image_name}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param,
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            self.drawing = True
            self.start_x = x
            self.start_y = y

        elif (
            event == cv2.EVENT_MOUSEMOVE
            and self.drawing
        ):

            self.draw()

            cv2.rectangle(
                self.display,
                (self.start_x, self.start_y),
                (x, y),
                (0, 255, 255),
                2,
            )

        elif event == cv2.EVENT_LBUTTONUP:

            self.drawing = False

            x1 = min(self.start_x, x)
            y1 = min(self.start_y, y)
            x2 = max(self.start_x, x)
            y2 = max(self.start_y, y)

            if x2 <= x1 or y2 <= y1:
                return

            image_height, image_width = self.image.shape[:2]

            x_center = (
                ((x1 + x2) / 2)
                / image_width
            )

            y_center = (
                ((y1 + y2) / 2)
                / image_height
            )

            width = (
                (x2 - x1)
                / image_width
            )

            height = (
                (y2 - y1)
                / image_height
            )

            self.boxes.append(
                (
                    self.current_class,
                    x_center,
                    y_center,
                    width,
                    height,
                )
            )

            self.draw()

    def run(self):

        window_name = "SIH YOLO Annotator"

        cv2.namedWindow(window_name)

        cv2.setMouseCallback(
            window_name,
            self.mouse_callback,
        )

        self.load_image()

        while True:

            self.draw()

            cv2.imshow(
                window_name,
                self.display,
            )

            key = cv2.waitKey(20) & 0xFF

            if key == ord("1"):

                self.current_class = 0
                print("Selected: blue_box")

            elif key == ord("2"):

                self.current_class = 1
                print("Selected: black_box")

            elif key == ord("d"):

                if self.boxes:
                    self.boxes.pop()
                    print("Removed last box.")

            elif key == ord("s"):

                self.save_labels()

                print(
                    f"Saved: {self.label_path()}"
                )

                self.index += 1

                if self.index >= len(self.image_paths):

                    print("Annotation complete.")
                    break

                self.load_image()

            elif key == ord("q"):

                print("Annotation stopped.")
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    annotator = YOLOAnnotator()
    annotator.run()

