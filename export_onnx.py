from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="onnx",
    imgsz=512,
    opset=12,
    simplify=True
)

print("Exported to best.onnx")