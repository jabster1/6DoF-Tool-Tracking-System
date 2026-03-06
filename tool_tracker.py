import cv2
from ultralytics import YOLO
from datetime import datetime
import time

# Load pre-trained model
model = YOLO("yolo11n.pt") 

# Define your shadowboard zone [x1, y1, x2, y2]
SHADOWBOARD_ZONE = [100, 100, 900, 700]

# Track tool states
tool_states = {}
last_log_time = {}  # Track when we last logged each tool
LOG_INTERVAL = 15  # seconds
LOG_FILE = "tool_log.txt"
MIN_CONFIDENCE = 0.5
CLASS_NAMES = {67: "Cell Phone", 76: "Scissors"}

CALIBRATION_MODE = True  # Set to False after calibration
EXPECTED_TOOLS = {}  # Will store what tools should be present
MISSING_CHECK_INTERVAL = 5  # Check for missing tools every 5 seconds
last_missing_check = 0

def log_event(tool_id, status, timestamp, class_name):
    current_time = time.time()
    
    # Only log if 15 seconds have passed since last log for this tool
    if tool_id not in last_log_time or (current_time - last_log_time[tool_id]) >= LOG_INTERVAL:
        message = f"[{timestamp}] {class_name} (ID: {tool_id}): {status}"
        
        # Print to console
        print(message)
        
        # Write to file
        with open(LOG_FILE, 'a') as f:
            f.write(message + '\n')
        
        last_log_time[tool_id] = current_time

def calibrate_tools(detected_tools):
    """Store the current detected tools as the expected baseline"""
    global EXPECTED_TOOLS
    EXPECTED_TOOLS = detected_tools.copy()
    print(f"\n=== CALIBRATION COMPLETE ===")
    print(f"Registered {len(EXPECTED_TOOLS)} tools:")
    for tool_id, info in EXPECTED_TOOLS.items():
        print(f"  - {info['class_name']} (ID: {tool_id})")
    print("Now monitoring for missing tools...\n")
    
    # Write to log file
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n=== Calibration: {len(EXPECTED_TOOLS)} tools registered ===\n")
        for tool_id, info in EXPECTED_TOOLS.items():
            f.write(f"  - {info['class_name']} (ID: {tool_id})\n")

def check_missing_tools(currently_detected, timestamp):
    """Check if any expected tools are missing from the zone"""
    for tool_id, info in EXPECTED_TOOLS.items():
        if tool_id not in currently_detected:
            # Tool is missing
            if tool_states.get(tool_id) != "MISSING":
                tool_states[tool_id] = "MISSING"
                log_event(tool_id, "MISSING FROM ZONE", timestamp, info['class_name'])

cap = cv2.VideoCapture(0)

print("Starting camera... Press 'q' to quit")
print("Looking for: Scissors (class 76) and Cell Phones (class 67)")
print(f"Logging to file: {LOG_FILE}")

if CALIBRATION_MODE:
    print("\n*** CALIBRATION MODE ***")
    print("Place all tools in the shadowboard zone, then press 'c' to calibrate")
else:
    print("\n*** MONITORING MODE ***")

# Clear/create the log file at start
with open(LOG_FILE, 'w') as f:
    f.write(f"=== Tool Tracking Log Started at {datetime.now()} ===\n\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO tracking
    results = model.track(frame, persist=True, conf=MIN_CONFIDENCE, verbose=False, 
                      tracker="bytetrack.yaml", iou=0.5)    #verbose false to take out extra output
    #tracker to try to keep same id when reenter frame same tool
    #adjust confidence to higher to not detect everything
    # Draw the shadowboard zone (blue rectangle)
    cv2.rectangle(frame, 
                  (SHADOWBOARD_ZONE[0], SHADOWBOARD_ZONE[1]), 
                  (SHADOWBOARD_ZONE[2], SHADOWBOARD_ZONE[3]), 
                  (255, 0, 0), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        currently_detected_in_zone = {}  # Track what's currently in zone this frame

        for box, cls, track_id, conf in zip(boxes, class_ids, track_ids, confidences):            # Look for scissors (76) or cell phone (67) as tool proxies
            if cls in [76, 67]:
                x1, y1, x2, y2 = box
                
                # Calculate center point of detected object
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Check if center is inside shadowboard zone
                is_in_zone = (SHADOWBOARD_ZONE[0] < cx < SHADOWBOARD_ZONE[2] and 
                              SHADOWBOARD_ZONE[1] < cy < SHADOWBOARD_ZONE[3])
                
                new_status = "IN" if is_in_zone else "OUT"
                class_name = CLASS_NAMES.get(cls, f"Class_{cls}")
                # If in calibration mode, just track tools in zone
                if CALIBRATION_MODE and is_in_zone:
                    currently_detected_in_zone[track_id] = {
                        'class_name': class_name,
                        'position': (cx, cy)
                    }

                # FIXED LOGIC: Only log when status changes AND enough time has passed
                # Only log if NOT in calibration mode
                if not CALIBRATION_MODE:
                    if track_id not in tool_states:
                        tool_states[track_id] = new_status
                        log_event(track_id, f"DETECTED {new_status}", datetime.now(), class_name)
                    elif tool_states[track_id] != new_status and tool_states[track_id] != "MISSING":
                        current_time = time.time()
                        if track_id not in last_log_time or (current_time - last_log_time[track_id]) >= LOG_INTERVAL:
                            tool_states[track_id] = new_status
                            log_event(track_id, f"CHANGED TO {new_status}", datetime.now(), class_name)
                    
                    # Track that this tool was detected in zone
                    if is_in_zone:
                        currently_detected_in_zone[track_id] = {
                            'class_name': class_name,
                            'position': (cx, cy)
                        }

                # Visual feedback
                status_text = f"{class_name} - {'IN PLACE' if is_in_zone else 'CHECKED OUT'}"
                color = (0, 255, 0) if is_in_zone else (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{status_text} (ID:{track_id})", 
                            (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.putText(frame, f"Conf: {conf:.2f}", 
                    (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Check for missing tools periodically (only when not calibrating)
    if not CALIBRATION_MODE:
        current_time = time.time()
        if current_time - last_missing_check >= MISSING_CHECK_INTERVAL:
            check_missing_tools(currently_detected_in_zone, datetime.now())
            last_missing_check = current_time

    cv2.imshow("Tool Tracker MVP", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c") and CALIBRATION_MODE:
        # Calibrate with currently detected tools
        if currently_detected_in_zone:
            calibrate_tools(currently_detected_in_zone)
            CALIBRATION_MODE = False
        else:
            print("No tools detected in zone! Place tools and try again.")

cap.release()
cv2.destroyAllWindows()

# Write closing message to log
with open(LOG_FILE, 'a') as f:
    f.write(f"\n=== Tool Tracking Stopped at {datetime.now()} ===\n")

print(f"System stopped. Check {LOG_FILE} for full log.")
