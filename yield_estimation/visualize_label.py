import cv2
import os

IMG_DIR = "reduced_dataset/train/images"
LABEL_DIR = "reduced_dataset/train/labels"
OUT_DIR = "reduced_dataset/label_viz"

os.makedirs(OUT_DIR, exist_ok=True)

for img_name in os.listdir(IMG_DIR):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.split())

                # YOLO → pixel coordinates
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    "orange",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

    out_path = os.path.join(OUT_DIR, img_name)
    cv2.imwrite(out_path, img)

print("Label visualization saved to:", OUT_DIR)
