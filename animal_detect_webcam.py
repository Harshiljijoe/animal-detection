import cv2
import torch
import pyttsx3
import time
import threading

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set up text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Start webcam
cap = cv2.VideoCapture(0)

# List of animal and person classes to detect
detect_classes = ['person', 'cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
last_alert_time = 0
alert_interval = 5  # seconds

def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 5 != 0:
        continue  # Skip some frames for performance

    results = model(frame)
    detected = []

    for *box, conf, cls in results.xyxy[0]:
        class_name = model.names[int(cls)]
        if class_name in detect_classes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            detected.append(class_name)

    current_time = time.time()
    if detected and current_time - last_alert_time > alert_interval:
        unique_detected = list(set(detected))
        alert_text = "Alert! " + ', '.join(unique_detected) + " detected"
        speak(alert_text)
        last_alert_time = current_time

    cv2.putText(frame, f"Detected: {len(detected)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Object Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
