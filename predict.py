"""
predict.py
----------
Simple script to test the trained YOLOv26 Nano model on any flower image, folder, or video.

Example usage in terminal:
    python predict.py --source test_flower.jpg --weights weights/best.pt
"""

import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Test YOLOv26n flower detector on images or videos.")
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='Path to trained best.pt file')
    parser.add_argument('--source', type=str, required=True, help='Path to an image, folder of images, or video')
    parser.add_argument('--conf', type=float, default=0.50, help='Minimum confidence threshold (default: 0.50)')
    args = parser.parse_args()

    # Load our trained model weights
    model = YOLO(args.weights)

    # Run prediction and save the output image with bounding boxes
    results = model.predict(
        source=args.source,
        imgsz=640,
        conf=args.conf,
        max_det=350,
        save=True
    )
    print(f"Done! Output images with bounding boxes are saved in: {results[0].save_dir}")

if __name__ == "__main__":
    main()