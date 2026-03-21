import os
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# Dataset path
DATA_YAML = "datasets/industrial-tools-v1/data.yaml"

# Load pretrained YOLOv11 medium checkpoint
model = YOLO("yolo11m.pt")

# Fine-tune on merged industrial tools dataset
results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=512,
    batch=16,
    lr0=0.001,
    freeze=10,
    device="mps",
    project="runs/train",
    name="industrial-tools-v1",
    pretrained=True,
    verbose=True
)

print("Training complete.")
print(f"Best model saved to: runs/train/industrial-tools-v1/weights/best.pt")