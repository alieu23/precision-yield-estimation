import os
import json
import random
import shutil
from collections import defaultdict

random.seed(42)

RAW_ROOT = "./orange_dataset"
OUT_ROOT = "./reduced_dataset"

SPLITS = {
    "train": 0.2, # 20% of original dataset for train
    "val": 0.05,    # 5% of original dataset for validation
    "test": 0.1    #10% of original dataset for training
}

# Create output dirs
for split in SPLITS:
    os.makedirs(f"{OUT_ROOT}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUT_ROOT}/{split}/labels", exist_ok=True)

categories = None
cat_id_to_yolo = {}

for condition in os.listdir(RAW_ROOT):
    cond_path = os.path.join(RAW_ROOT, condition)

    if not os.path.isdir(cond_path):
        continue

    with open(os.path.join(cond_path, "annotations.json")) as f:
        coco = json.load(f)

    if categories is None:
        categories = coco["categories"]
        cat_id_to_yolo = {c["id"]: i for i, c in enumerate(categories)}

    images = coco["images"]
    print("number of images: ", len(images))
    annotations = coco["annotations"]

    random.shuffle(images)

    n = len(images)

    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    n_test = int(n * SPLITS["test"])

    split_imgs = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:n_train + n_val + n_test]
    }

    # Map image_id -> annotations
    ann_map = defaultdict(list)
    for ann in annotations:
        ann_map[ann["image_id"]].append(ann)

    for split, imgs in split_imgs.items():
        for img in imgs:
            img_id = img["id"]
            w, h = img["width"], img["height"]

            # Copy image
            src_img = os.path.join(cond_path, "images", img["file_name"])
            dst_img = os.path.join(OUT_ROOT, split, "images", img["file_name"])
            shutil.copy(src_img, dst_img)

            label_path = os.path.join(
                OUT_ROOT, split, "labels",
                img["file_name"].replace(".jpg", ".txt")
            )

            with open(label_path, "w") as f:
                for ann in ann_map[img_id]:
                    x, y, bw, bh = ann["bbox"]

                    # COCO → YOLO
                    x_center = (x + bw / 2) / w
                    y_center = (y + bh / 2) / h
                    bw /= w
                    bh /= h

                    cls = cat_id_to_yolo[ann["category_id"]]

                    f.write(f"{cls} {x_center} {y_center} {bw} {bh}\n")

print("Images and labels successfully created")
