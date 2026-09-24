from pathlib import Path
from PIL import Image, ImageDraw

INPUT_DIR = Path("data/inspection/video5")
OUTPUT_FILE = Path("data/inspection/video5_contact_sheet.jpg")

files = sorted(INPUT_DIR.glob("*.jpg"))

images = [Image.open(f).convert("RGB") for f in files]

thumb_w = 320
thumb_h = 240
cols = 3
rows = (len(images) + cols - 1) // cols

sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
draw = ImageDraw.Draw(sheet)

for i, (img, file) in enumerate(zip(images, files)):
    img.thumbnail((thumb_w, thumb_h))

    x = (i % cols) * thumb_w
    y = (i // cols) * thumb_h

    sheet.paste(img, (x, y))
    draw.text((x + 5, y + 5), file.stem, fill="red")

sheet.save(OUTPUT_FILE, quality=90)

print(f"Saved: {OUTPUT_FILE}")