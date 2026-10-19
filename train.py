"""
train.py
--------
Trains the YOLOv26 Nano (yolo26n.pt) model on our curated 5-class flower dataset.
Hardware-aware hyperparameters set to prevent Linux out-of-memory crashes.
"""

from ultralytics import YOLO

def main():
    # Load the pretrained YOLOv26 Nano model
    model = YOLO('yolo26n.pt')

    # Start training
    model.train(
        data='/workspace/curated_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=32,
        workers=2,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        patience=30,
        max_det=350,
        device=0,
        project='/workspace/runs',
        name='yolov26n_curated_clean',
        exist_ok=True
    )

if __name__ == "__main__":
    main()