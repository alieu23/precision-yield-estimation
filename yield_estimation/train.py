from ultralytics import YOLO
from pathlib import Path


def train_model():
    DATA_YAML = Path("reduced_dataset/data.yaml")
    PROJECT_DIR = "yield_estimation_project"
    EXP_NAME = "orange_detection_v1"
    # Load the YOLOv11 nano model for efficiency

    model = YOLO('yolo11n.pt')

    model.train(
        data = str(DATA_YAML),
        epochs = 100,
        patience=50,
        imgsz = 640,
        batch = 16,
        project=PROJECT_DIR,
        name=EXP_NAME,
        device = 'cpu',
        mosaic = 1.0,
        mixup = 0.2
    )

if __name__ == '__main__':
        train_model()