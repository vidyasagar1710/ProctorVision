
import streamlit as st
import cv2
import time
import os
import tempfile
import numpy as np
import pandas as pd

from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ProctorVision | Team 22",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PROFESSIONAL DARK MODE CSS
# =========================================================

st.markdown("""
<style>

/* Overall website */

.stApp {
    background-color: #080F1E;
    color: #F8FAFC;
}

.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 100%;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #111C30;
    border-right: 1px solid #293B56;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

/* Headings */

h1, h2, h3, h4 {
    color: #F8FAFC !important;
}

p, label {
    color: #CBD5E1;
}

/* Buttons */

.stButton > button {
    border-radius: 8px;
    height: 45px;
    font-weight: 600;
    transition: 0.2s;
}

/* Start button */

.stButton > button[kind="primary"] {
    background-color: #16A34A;
    color: white !important;
    border: none;
}

.stButton > button[kind="primary"]:hover {
    background-color: #15803D;
    border: 1px solid #4ADE80;
}

/* Stop button */

.stButton > button[kind="secondary"] {
    background-color: #DC2626;
    color: white !important;
    border: none;
}

.stButton > button[kind="secondary"]:hover {
    background-color: #991B1B;
}

/* Metric cards */

[data-testid="stMetric"] {
    background-color: #17243A;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #293B56;
}

[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
}

/* Alert messages */

[data-testid="stAlert"] {
    background-color: #17243A !important;
    border: 1px solid #293B56;
    border-radius: 10px;
}

[data-testid="stAlert"] p {
    color: #CBD5E1 !important;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #17243A;
    border: 1px solid #293B56;
    border-radius: 10px;
    padding: 10px;
}

/* Input fields */

.stTextInput input {
    background-color: #17243A;
    color: white;
    border: 1px solid #334155;
}

/* Tables */

[data-testid="stDataFrame"] {
    background-color: #17243A;
    border-radius: 10px;
}

/* Alert cards */

.alert-card {
    background-color: #17243A;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #EF4444;
    border-top: 1px solid #293B56;
    border-right: 1px solid #293B56;
    border-bottom: 1px solid #293B56;
    margin-bottom: 12px;
    color: #F8FAFC;
    line-height: 1.8;
}

/* Normal cards */

.normal-card {
    background-color: #17243A;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #22C55E;
    margin-bottom: 12px;
    color: #F8FAFC;
}

/* Dividers */

hr {
    border-color: #293B56;
}

/* Download button */

.stDownloadButton button {
    background-color: #2563EB;
    color: white !important;
    border-radius: 8px;
    border: none;
    height: 42px;
}

.stDownloadButton button:hover {
    background-color: #1D4ED8;
}

/* Select box */

[data-baseweb="select"] {
    background-color: #17243A;
}

/* Video image */

[data-testid="stImage"] {
    border: 1px solid #293B56;
    border-radius: 10px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. PROCTORVISION HEADER
# =========================================================

st.html("""
<div style="
    background: linear-gradient(120deg, #14243D, #0B1629);
    padding: 20px 28px;
    border-radius: 14px;
    border: 1px solid #293B56;
    margin-bottom: 22px;
">

    <h1 style="
        color: #38BDF8;
        font-size: 34px;
        margin: 0;
        font-weight: 700;
    ">
        🛡️ ProctorVision
    </h1>

    <p style="
        color: #CBD5E1;
        font-size: 15px;
        margin: 8px 0;
    ">
        AI-Powered Automated Examination
        Malpractice Detection System
    </p>

    <p style="
        color: #64748B;
        font-size: 12px;
        margin: 0;
    ">
        TEAM 22 | INTELLIGENT EXAMINATION SURVEILLANCE
    </p>

</div>
""")


# =========================================================
# 4. LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():

    return YOLO("yolo11n-pose.pt")


# =========================================================
# 5. SESSION STATE INITIALIZATION
# =========================================================

defaults = {
    "running": False,
    "cap": None,
    "writer": None,
    "model": None,
    "alerts": [],
    "total_alerts": 0,
    "frame_number": 0,
    "fps": 25.0,
    "suspicious_frames": defaultdict(int),
    "last_alert": defaultdict(lambda: -999),
    "video_temp": None,
    "output_temp": None,
    "detected": 0,
    "suspicious_count": 0,
    "current_students": [],
    "source_name": "Not connected",
    "confidence": 0.40,
    "alert_duration": 2.0,
    "cooldown": 5.0
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 6. RELEASE VIDEO RESOURCES
# =========================================================

def release_video():

    if st.session_state.cap is not None:

        st.session_state.cap.release()
        st.session_state.cap = None

    if st.session_state.writer is not None:

        st.session_state.writer.release()
        st.session_state.writer = None

    st.session_state.running = False


# =========================================================
# 7. SIDEBAR SETTINGS
# =========================================================

st.sidebar.markdown("""
<h2 style="color:#38BDF8;">
    ⚙️ Monitoring Settings
</h2>
""", unsafe_allow_html=True)

source_type = st.sidebar.selectbox(
    "Select Video Source",
    ["Upload Video", "Webcam", "CCTV RTSP"]
)

uploaded_video = None
rtsp_url = ""

if source_type == "Upload Video":

    uploaded_video = st.sidebar.file_uploader(
        "Upload Examination Video",
        type=["mp4", "avi", "mov"]
    )

elif source_type == "CCTV RTSP":

    rtsp_url = st.sidebar.text_input(
        "CCTV RTSP URL",
        placeholder="rtsp://username:password@camera-ip/stream"
    )


st.sidebar.markdown("---")

st.sidebar.markdown("### Detection Configuration")

confidence_threshold = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05
)

alert_duration = st.sidebar.slider(
    "Alert Duration (seconds)",
    min_value=1.0,
    max_value=10.0,
    value=2.0,
    step=0.5
)

st.sidebar.markdown("---")


# =========================================================
# 8. START AND STOP BUTTONS
# =========================================================

start_button = st.sidebar.button(
    "▶ START MONITORING",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.running
)

stop_button = st.sidebar.button(
    "■ STOP MONITORING",
    type="secondary",
    use_container_width=True,
    disabled=not st.session_state.running
)


# =========================================================
# 9. START VIDEO SOURCE
# =========================================================

if start_button:

    try:

        # Release old resources

        release_video()

        st.session_state.alerts = []
        st.session_state.total_alerts = 0
        st.session_state.frame_number = 0

        st.session_state.suspicious_frames = defaultdict(int)
        st.session_state.last_alert = defaultdict(lambda: -999)

        st.session_state.confidence = confidence_threshold
        st.session_state.alert_duration = alert_duration

        # Load YOLO

        with st.spinner("Loading ProctorVision AI model..."):

            st.session_state.model = load_model()

        # Video source

        if source_type == "Upload Video":

            if uploaded_video is None:

                st.error("Please upload an examination video.")

                st.stop()

            temp = tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            )

            temp.write(uploaded_video.getvalue())
            temp.close()

            st.session_state.video_temp = temp.name

            cap = cv2.VideoCapture(temp.name)

            st.session_state.source_name = uploaded_video.name

        elif source_type == "Webcam":

            cap = cv2.VideoCapture(0)

            st.session_state.source_name = "Webcam 01"

        else:

            if not rtsp_url.strip():

                st.error("Please enter the CCTV RTSP URL.")

                st.stop()

            cap = cv2.VideoCapture(rtsp_url)

            st.session_state.source_name = "CCTV Camera 01"

        if not cap.isOpened():

            cap.release()

            st.error("Unable to connect to video source.")

            st.stop()

        st.session_state.cap = cap

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            fps = 25.0

        st.session_state.fps = fps

        # Prepare output video

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output = tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        )

        output.close()

        st.session_state.output_temp = output.name

        writer = cv2.VideoWriter(
            output.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        st.session_state.writer = writer

        st.session_state.running = True

        st.rerun()

    except Exception as e:

        release_video()

        st.error(f"Error starting monitoring: {e}")


# =========================================================
# 10. STOP VIDEO SOURCE
# =========================================================

if stop_button:

    release_video()

    st.rerun()


# =========================================================
# 11. VIDEO PROCESSING FUNCTION
# =========================================================

def process_frame():

    cap = st.session_state.cap

    if cap is None:
        return None

    ret, frame = cap.read()

    if not ret:

        release_video()

        return None

    st.session_state.frame_number += 1

    frame_number = st.session_state.frame_number

    fps = st.session_state.fps

    timestamp = frame_number / fps

    model = st.session_state.model

    # YOLO tracking

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=st.session_state.confidence,
        tracker="bytetrack.yaml",
        verbose=False
    )

    annotated = frame.copy()

    detected = 0
    suspicious_count = 0

    current_students = []

    # -----------------------------------------------------
    # STUDENT DETECTION
    # -----------------------------------------------------

    for result in results:

        if result.keypoints is None:
            continue

        if result.boxes.id is None:
            continue

        keypoints = result.keypoints.xy.cpu().numpy()

        ids = result.boxes.id.int().cpu().tolist()

        boxes = result.boxes.xyxy.cpu().numpy()

        confidences = result.boxes.conf.cpu().numpy()

        for i, student_id in enumerate(ids):

            points = keypoints[i]

            nose = points[0]
            left_eye = points[1]
            right_eye = points[2]

            # Validate facial landmarks

            if min(
                nose[0],
                left_eye[0],
                right_eye[0]
            ) <= 0:

                continue

            eye_distance = np.linalg.norm(
                left_eye - right_eye
            )

            if eye_distance < 5:
                continue

            detected += 1

            # -------------------------------------------------
            # HEAD DIRECTION ESTIMATION
            # -------------------------------------------------

            eye_center = (
                left_eye[0] + right_eye[0]
            ) / 2

            deviation = (
                nose[0] - eye_center
            ) / eye_distance

            if deviation < -0.25:

                direction = "LEFT"

            elif deviation > 0.25:

                direction = "RIGHT"

            else:

                direction = "CENTRE"

            suspicious = direction != "CENTRE"

            if suspicious:

                st.session_state.suspicious_frames[student_id] += 1

            else:

                st.session_state.suspicious_frames[student_id] = 0

            duration = (
                st.session_state.suspicious_frames[student_id] / fps
            )

            confidence = float(confidences[i])

            alert = (
                duration >= st.session_state.alert_duration
            )

            # -------------------------------------------------
            # ALERT CLASSIFICATION
            # -------------------------------------------------

            if alert:

                suspicious_count += 1

                color = (0, 0, 255)

                level = (
                    "HIGH" if duration >= 5
                    else "MEDIUM"
                )

                if (
                    timestamp - st.session_state.last_alert[student_id]
                    >= st.session_state.cooldown
                ):

                    st.session_state.total_alerts += 1

                    st.session_state.last_alert[student_id] = timestamp

                    record = {

                        "Student ID": student_id,

                        "Time": datetime.now().strftime("%H:%M:%S"),

                        "Timestamp": round(timestamp, 2),

                        "Behaviour": "Sideways head orientation",

                        "Direction": direction,

                        "Duration": round(duration, 2),

                        "Detection Confidence": round(confidence, 2),

                        "Alert Level": level

                    }

                    st.session_state.alerts.insert(0, record)

            else:

                color = (0, 255, 0)

                level = "NORMAL"

            # -------------------------------------------------
            # STUDENT BOUNDING BOX
            # -------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                boxes[i]
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Information panel

            labels = [

                f"ID: {student_id} | {level}",

                f"Head: {direction}",

                f"Duration: {duration:.1f}s",

                f"Confidence: {confidence:.2f}"

            ]

            panel_height = 82

            top = max(0, y1 - panel_height)

            cv2.rectangle(
                annotated,
                (x1, top),
                (x2, y1),
                color,
                -1
            )

            for j, label in enumerate(labels):

                cv2.putText(
                    annotated,
                    label,
                    (
                        x1 + 5,
                        max(15, y1 - 65 + j * 18)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1
                )

            # Draw facial landmarks

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

            current_students.append({

                "Student ID": student_id,

                "Head Direction": direction,

                "Duration": round(duration, 1),

                "Detection Confidence": round(confidence, 2),

                "Status": level

            })

    # -----------------------------------------------------
    # VIDEO HEADER OVERLAY
    # -----------------------------------------------------

    cv2.rectangle(
        annotated,
        (0, 0),
        (annotated.shape[1], 75),
        (25, 35, 55),
        -1
    )

    cv2.putText(
        annotated,
        "PROCTORVISION | AI MONITORING",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"Students: {detected} | Alerts: {st.session_state.total_alerts}",
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    if suspicious_count > 0:

        cv2.putText(
            annotated,
            "SUSPICIOUS ACTIVITY DETECTED - VERIFY",
            (15, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

    # Save processed video

    if st.session_state.writer is not None:

        st.session_state.writer.write(annotated)

    st.session_state.detected = detected

    st.session_state.suspicious_count = suspicious_count

    st.session_state.current_students = current_students

    return annotated


# =========================================================
# 12. DASHBOARD DISPLAY
# =========================================================

@st.fragment(run_every=0.15)
def monitoring_dashboard():

    left, right = st.columns(
        [1, 2],
        gap="medium"
    )

    # =====================================================
    # LEFT PANEL - ALERTS
    # =====================================================

    with left:

        st.markdown("## 🚨 LIVE ALERTS")

        if st.session_state.running:

            annotated = process_frame()

        else:

            annotated = None

        if st.session_state.alerts:

            for alert in st.session_state.alerts[:10]:

                st.html(f"""
                <div class="alert-card">

                    <b style="color:#F87171;">
                        🚨 Student ID: {alert['Student ID']}
                    </b>

                    <br><br>

                    <b>Behaviour:</b> {alert['Behaviour']}<br>

                    <b>Head Direction:</b> {alert['Direction']}<br>

                    <b>Duration:</b> {alert['Duration']} seconds<br>

                    <b>Detection Confidence:</b>
                    {alert['Detection Confidence']}<br>

                    <b>Alert Level:</b> {alert['Alert Level']}<br>

                    <b>Time:</b> {alert['Time']}<br>

                    <b>Video Timestamp:</b> {alert['Timestamp']}s

                    <br><br>

                    <span style="color:#FBBF24;">
                        Awaiting invigilator verification
                    </span>

                </div>
                """)

        else:

            st.html("""
            <div class="normal-card">
                🟢 No suspicious activities detected.
                <br>
                <span style="color:#94A3B8;">
                    Alerts will appear here during monitoring.
                </span>
            </div>
            """)

        st.markdown("---")

        # =================================================
        # STATISTICS
        # =================================================

        st.markdown("## 📊 MONITORING STATISTICS")

        m1, m2 = st.columns(2)

        with m1:

            st.metric(
                "Students Detected",
                st.session_state.detected
            )

        with m2:

            st.metric(
                "Suspicious Students",
                st.session_state.suspicious_count
            )

        st.metric(
            "Total Alerts",
            st.session_state.total_alerts
        )

        st.markdown("---")

        # =================================================
        # INCIDENT HISTORY
        # =================================================

        st.markdown("## 📋 INCIDENT HISTORY")

        if st.session_state.alerts:

            history = pd.DataFrame(
                st.session_state.alerts
            )

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

            csv_data = history.to_csv(index=False)

            st.download_button(
                "📥 DOWNLOAD ALERT REPORT",
                csv_data,
                "ProctorVision_Alert_Report.csv",
                "text/csv",
                use_container_width=True
            )

        else:

            st.caption("No incidents recorded yet.")

    # =====================================================
    # RIGHT PANEL - CCTV VIDEO
    # =====================================================

    with right:

        st.markdown("## 📹 LIVE CCTV VIDEO")

        if annotated is not None:

            st.image(
                cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                ),
                channels="RGB",
                use_container_width=True
            )

        elif st.session_state.running:

            st.info("Waiting for video frames...")

        else:

            st.html("""
            <div style="
                background:#17243A;
                border:1px solid #293B56;
                border-radius:12px;
                height:360px;
                display:flex;
                flex-direction:column;
                align-items:center;
                justify-content:center;
                text-align:center;
            ">

                <div style="font-size:55px;">📹</div>

                <h3 style="color:#38BDF8;">
                    CCTV Video Monitoring
                </h3>

                <p style="color:#94A3B8;">
                    Select a video source and click START MONITORING.
                </p>

                <p style="color:#64748B;font-size:12px;">
                    PROCTORVISION | TEAM 22
                </p>

            </div>
            """)

        st.markdown("---")

        # =================================================
        # STUDENT DETECTION DETAILS
        # =================================================

        st.markdown("## 🎯 STUDENT DETECTION DETAILS")

        if st.session_state.current_students:

            students = pd.DataFrame(
                st.session_state.current_students
            )

            st.dataframe(
                students,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.caption("Waiting for student detection...")

        # =================================================
        # SYSTEM STATUS
        # =================================================

        st.markdown("---")

        st.markdown("## ⚙️ SYSTEM STATUS")

        if st.session_state.running:

            st.success("AI Monitoring Active")

        else:

            st.info("Monitoring is currently stopped.")

        st.caption(
            f"Video Source: {st.session_state.source_name}"
        )


# =========================================================
# 13. START DASHBOARD
# =========================================================

monitoring_dashboard()


# =========================================================
# 14. VIDEO COMPLETION MESSAGE
# =========================================================

if (
    not st.session_state.running
    and st.session_state.frame_number > 0
    and st.session_state.cap is None
):

    st.success("Video monitoring completed or stopped.")

    if st.session_state.output_temp and os.path.exists(
        st.session_state.output_temp
    ):

        st.download_button(
            "📹 DOWNLOAD PROCESSED VIDEO",
            data=open(st.session_state.output_temp, "rb").read(),
            file_name="ProctorVision_Processed_Video.mp4",
            mime="video/mp4"
        )

