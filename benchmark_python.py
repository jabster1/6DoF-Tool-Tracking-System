import time
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Open webcam or video file
cap = cv2.VideoCapture(0)  # 0 = webcam

frame_times = []
num_frames = 100  # benchmark over 100 frames

print("Running Python inference benchmark...")

for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    start = time.perf_counter()
    results = model(frame, imgsz=512, verbose=False)
    end = time.perf_counter()
    
    frame_times.append((end - start) * 1000)  # convert to ms

cap.release()

avg_time = sum(frame_times) / len(frame_times)
fps = 1000 / avg_time

print(f"Average inference time: {avg_time:.2f}ms per frame")
print(f"Average FPS: {fps:.1f}")
print(f"Min: {min(frame_times):.2f}ms  Max: {max(frame_times):.2f}ms")