import cv2
import os
import math
import threading
import time
from ultralytics import YOLO

# Force OpenCV to use TCP for RTSP stream to prevent grey screen/packet loss
#os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Load YOLO model
model = YOLO("yolov8n.pt")

class BackgroundRTSPReader:
    """ 
    Reads frames in a separate background thread. 
    This prevents OpenCV's buffer from overflowing and causing grey frames 
    when YOLO inference takes too long!
    """
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        # Using buffer size 1 ensures we only ever hold the absolute latest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                # Continuously pull frames to keep the buffer completely empty
                ret, frame = self.cap.read()
                if ret:
                    self.ret = ret
                    self.frame = frame.copy() # Store a fresh copy
            time.sleep(0.01)

    def read(self):
        return self.ret, self.frame
    
    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

# Your RTSP stream
#rtsp_url = "rtsp://admin:System@123@192.168.1.107:554/stream1"

# Use the background reader instead of plain cv2.VideoCapture
#cap = BackgroundRTSPReader(rtsp_url)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("Waiting for streams/connection...")
        time.sleep(1)
        continue

    # Actively reduce resolution to speed up YOLO processing and prevent packet build-up
    frame = cv2.resize(frame, (640, 480))

    results = model(frame)
    persons = []

    # First pass: collect all people
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Calculate center of the person
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                
                persons.append({"box": (x1, y1, x2, y2), "center": (cx, cy), "width": w, "conf": conf})

    # Second pass: Check for close proximity (basic fighting logic)
    fighting_alert = False
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            cx1, cy1 = persons[i]["center"]
            cx2, cy2 = persons[j]["center"]
            
            # Calculate distance between two people
            distance = math.hypot(cx1 - cx2, cy1 - cy2)
            
            # Use average width of the two people as a relative threshold
            avg_width = (persons[i]["width"] + persons[j]["width"]) / 2
            
            # If they are exceptionally close to each other
            if distance < avg_width * 0.7:
                fighting_alert = True
                
                # Draw red bounding boxes for the involved people
                x1_a, y1_a, x2_a, y2_a = persons[i]["box"]
                x1_b, y1_b, x2_b, y2_b = persons[j]["box"]
                cv2.rectangle(frame, (x1_a, y1_a), (x2_a, y2_a), (0, 0, 255), 3)
                cv2.rectangle(frame, (x1_b, y1_b), (x2_b, y2_b), (0, 0, 255), 3)

    # Draw regular green boxes for non-fighting people
    if not fighting_alert:
        for p in persons:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {p['conf']:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display Alert Message
    if fighting_alert:
        cv2.putText(frame, "ALERT: FIGHTING / CLOSE CONTACT!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()