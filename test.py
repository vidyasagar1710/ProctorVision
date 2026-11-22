
import cv2
import time
import csv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ---------------- CONFIGURATION ----------------

VIDEO_PATH = "exam_video.mp4"
OUTPUT_VIDEO = "malpractice_output.mp4"
ALERT_FILE = "malpractice_alerts.csv"

CONFIDENCE = 0.40
SUSPICIOUS_THRESHOLD = 0.25
ALERT_DURATION = 2.0
ALERT_COOLDOWN = 5.0

# ---------------- LOAD MODEL ----------------

model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Cannot open examination video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# ---------------- TRACKING DATA ----------------

suspicious_frames = defaultdict(int)
last_alert_time = defaultdict(lambda: -999)

frame_number = 0
total_alerts = 0
alert_history = []

# ---------------- CSV REPORT ----------------

with open(ALERT_FILE, "w", newline="") as file:

    csv_writer = csv.writer(file)

    csv_writer.writerow([
        "Student ID",
        "Timestamp",
        "Head Direction",
        "Suspicious Duration",
        "Confidence",
        "Alert Level"
    ])

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1
        timestamp = frame_number / fps

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONFIDENCE,
            tracker="bytetrack.yaml",
            verbose=False
        )

        annotated = frame.copy()

        detected_students = 0
        suspicious_students = 0

        # ---------------- STUDENT ANALYSIS ----------------

        for result in results:

            if result.keypoints is None or result.boxes.id is None:
                continue

            keypoints = result.keypoints.xy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for i, student_id in enumerate(track_ids):

                points = keypoints[i]

                nose = points[0]
                left_eye = points[1]
                right_eye = points[2]

                if min(nose[0], left_eye[0],
                       right_eye[0]) <= 0:
                    continue

                eye_distance = np.linalg.norm(
                    left_eye - right_eye
                )

                if eye_distance < 5:
                    continue

                detected_students += 1

                eye_center_x = (
                    left_eye[0] + right_eye[0]
                ) / 2

                deviation = (
                    nose[0] - eye_center_x
                ) / eye_distance

                # Approximate head direction
                if deviation < -SUSPICIOUS_THRESHOLD:
                    direction = "LEFT"
                elif deviation > SUSPICIOUS_THRESHOLD:
                    direction = "RIGHT"
                else:
                    direction = "CENTRE"

                suspicious = direction != "CENTRE"

                if suspicious:
                    suspicious_frames[student_id] += 1
                else:
                    suspicious_frames[student_id] = 0

                duration = suspicious_frames[student_id] / fps

                x1, y1, x2, y2 = map(int, boxes[i])
                confidence = float(confidences[i])

                alert = duration >= ALERT_DURATION

                if alert:
                    suspicious_students += 1

                    alert_level = (
                        "HIGH" if duration >= 5 else "MEDIUM"
                    )

                    color = (0, 0, 255)

                    if timestamp - last_alert_time[student_id] >= ALERT_COOLDOWN:

                        total_alerts += 1
                        last_alert_time[student_id] = timestamp

                        alert_history.append(
                            f"Student {student_id}: {direction} at {timestamp:.1f}s"
                        )

                        csv_writer.writerow([
                            student_id,
                            round(timestamp, 2),
                            direction,
                            round(duration, 2),
                            round(confidence, 3),
                            alert_level
                        ])

                        file.flush()

                else:
                    alert_level = "NORMAL"
                    color = (0, 255, 0)

                # ---------------- BOUNDING BOX ----------------

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Information above bounding box
                cv2.rectangle(
                    annotated,
                    (x1, max(0, y1 - 85)),
                    (x2, y1),
                    color,
                    -1
                )

                details = [
                    f"ID: {student_id} | {alert_level}",
                    f"Head: {direction}",
                    f"Duration: {duration:.1f}s",
                    f"Confidence: {confidence:.2f}"
                ]

                for j, text in enumerate(details):

                    cv2.putText(
                        annotated,
                        text,
                        (x1 + 5, max(15, y1 - 67 + j * 17)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        (255, 255, 255),
                        1
                    )

                # Draw detected facial landmarks
                for point in points[:5]:

                    px, py = map(int, point)

                    if px > 0 and py > 0:
                        cv2.circle(
                            annotated,
                            (px, py),
                            3,
                            (255, 255, 0),
                            -1
                        )

        # ---------------- DASHBOARD PANEL ----------------

        cv2.rectangle(
            annotated,
            (0, 0),
            (width, 105),
            (35, 35, 35),
            -1
        )

        dashboard = [
            "TEAM 22 - AI EXAMINATION MONITORING",
            f"TIME: {timestamp:.1f}s",
            f"STUDENTS DETECTED: {detected_students}",
            f"SUSPICIOUS STUDENTS: {suspicious_students}",
            f"TOTAL ALERTS: {total_alerts}"
        ]

        for j, text in enumerate(dashboard):

            cv2.putText(
                annotated,
                text,
                (15, 22 + j * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255),
                1
            )

        # ---------------- LIVE ALERT ----------------

        if suspicious_students > 0:

            cv2.putText(
                annotated,
                "WARNING: SUSPICIOUS BEHAVIOUR DETECTED",
                (15, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                annotated,
                "STATUS: NORMAL MONITORING",
                (15, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        # ---------------- SAVE AND DISPLAY ----------------

        writer.write(annotated)

        cv2.imshow(
            "TEAM 22 - MALPRACTICE DETECTION",
            annotated
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("\nEXAMINATION ANALYSIS COMPLETED")
print("Total alerts:", total_alerts)
print("Output video:", OUTPUT_VIDEO)
print("Alert report:", ALERT_FILE)
