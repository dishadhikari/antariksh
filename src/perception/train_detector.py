from ultralytics import YOLO


def main():
    model = YOLO("models/detector/best.pt")

    model.train(
        data="data/detection/dataset.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        project="runs/detection",
        name="box_detector_v2",
    )


if __name__ == "__main__":
    main()