import os
os.environ["YOLO_VERBOSE"] = "False"

import cv2
import math
import time
import numpy as np
import sounddevice as sd
from ultralytics import YOLO
from collections import deque

# =========================
# CONFIG
# =========================
CONF_THRESHOLD = 0.5
DIST_THRESHOLD = 0.5
MOTION_THRESHOLD = 20
FIGHT_FRAMES = 10

# 🔊 AUDIO (FIXED)
AUDIO_THRESHOLD = 0.01   # LOWERED
AUDIO_FRAMES = 5         # FASTER RESPONSE

FRAME_SIZE = (640, 480)

# =========================
# MODEL
# =========================
model = YOLO("yolov8n.pt")

# =========================
# AUDIO SYSTEM (SIMPLIFIED)
# =========================
audio_buffer = deque(maxlen=30)

def audio_callback(indata, frames, time_info, status):
    volume = np.sqrt(np.mean(indata**2))  # RMS
    audio_buffer.append(volume)

# ✅ use correct mic (WASAPI)
stream = sd.InputStream(device=9, callback=audio_callback)
stream.start()

# =========================
# VIDEO INPUT
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

# =========================
# TRACKING VARIABLES
# =========================
prev_centers = []
fight_counter = 0
audio_counter = 0

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()

    if not ret:
        time.sleep(0.5)
        continue

    frame = cv2.resize(frame, FRAME_SIZE)

    # =========================
    # YOLO DETECTION
    # =========================
    results = model(frame, conf=CONF_THRESHOLD, imgsz=416, verbose=False)

    persons = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] != "person":
                continue

            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w = x2 - x1

            persons.append({
                "box": (x1, y1, x2, y2),
                "center": (cx, cy),
                "width": w
            })

    # =========================
    # FIGHT DETECTION
    # =========================
    fight_detected = False

    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            c1 = persons[i]["center"]
            c2 = persons[j]["center"]

            dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
            avg_w = (persons[i]["width"] + persons[j]["width"]) / 2

            close = dist < avg_w * DIST_THRESHOLD

            motion = 0
            if i < len(prev_centers):
                motion = math.hypot(
                    c1[0] - prev_centers[i][0],
                    c1[1] - prev_centers[i][1]
                )

            aggressive = motion > MOTION_THRESHOLD

            if close and aggressive:
                fight_counter += 1
            else:
                fight_counter = max(0, fight_counter - 1)

            if fight_counter > FIGHT_FRAMES:
                fight_detected = True

    prev_centers = [p["center"] for p in persons]

    # =========================
    # AUDIO DETECTION (FIXED)
    # =========================
    loud_audio = False
    avg_audio = 0

    if len(audio_buffer) > 0:
        avg_audio = np.mean(audio_buffer)

        # ✅ SIMPLE + WORKING
        if avg_audio > AUDIO_THRESHOLD:
            audio_counter += 1
        else:
            audio_counter = max(0, audio_counter - 1)

    loud_audio = audio_counter > AUDIO_FRAMES

    # =========================
    # DRAWING
    # =========================

    for p in persons:
        x1, y1, x2, y2 = p["box"]

        color = (0, 255, 0)
        if fight_detected:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # =========================
    # ALERT LOGIC
    # =========================
    alert_text = "SAFE"
    alert_color = (0, 255, 0)

    if fight_detected:
        alert_text = "FIGHT DETECTED"
        alert_color = (0, 0, 255)

    elif loud_audio:
        alert_text = "LOUD SHOUT DETECTED"
        alert_color = (255, 0, 0)

    # =========================
    # TOP BANNER
    # =========================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 80), alert_color, -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.putText(frame, alert_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # =========================
    # STATUS PANEL
    # =========================
    cv2.rectangle(frame, (10, 380), (260, 470), (50, 50, 50), -1)

    cv2.putText(frame, f"People: {len(persons)}", (20, 410),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Fight: {'YES' if fight_detected else 'NO'}", (20, 435),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if fight_detected else (0, 255, 0), 2)

    cv2.putText(frame, f"Audio: {'LOUD' if loud_audio else 'NORMAL'}", (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 0, 0) if loud_audio else (0, 255, 0), 2)

    # 🔊 DEBUG ON SCREEN (NOT TERMINAL)
    cv2.putText(frame, f"AudioLvl: {round(avg_audio,3)}", (300, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # =========================
    # DISPLAY
    # =========================
    cv2.imshow("CCTV AI System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
stream.stop()