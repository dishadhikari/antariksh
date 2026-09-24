import cv2
from pathlib import Path

VIDEO = Path("data/raw/box_experiment/videos/5.mp4")
OUTPUT = Path("data/inspection/video5")
OUTPUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for frame_no in range(0, total, 10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()

    if not ok:
        continue

    cv2.imwrite(
        str(OUTPUT / f"frame_{frame_no:04d}.jpg"),
        frame
    )

cap.release()

print(f"Saved frames to: {OUTPUT}")