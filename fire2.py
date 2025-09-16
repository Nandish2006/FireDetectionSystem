from ultralytics import YOLO
import cvzone
import cv2
import streamlit as st
import os
import time
from datetime import datetime
import pandas as pd
from twilio.rest import Client
import requests
import tempfile

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Fire Detection System", layout="wide")
st.title("🔥 Real-Time Fire Detection Dashboard")

# Sidebar controls
source = st.sidebar.radio("Select Source", ["Webcam", "Sample Video"])
start_button = st.sidebar.button("▶ Start Stream")
stop_button = st.sidebar.button("⏹ Stop Stream")

# Load YOLO model
model = YOLO("fire.pt")  # Make sure fire.pt is in the folder
classnames = ["fire"]

# Twilio config
TWILIO_ACCOUNT_SID = "ACa74fab25d55343e1134cb6570a2ca890"
TWILIO_AUTH_TOKEN = "5e96bc18f67a3c2158276a14f6ea2f5e"
TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"
TO_PHONE = "whatsapp:+918320437952"

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ImgBB config
IMGBB_API_KEY = "302bb55d23ed10f7a57549d24b674165"  # Replace with your key

def upload_to_imgbb(frame, api_key):
    """Uploads a frame to ImgBB and returns the public URL."""
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(temp_file.name, frame)
        with open(temp_file.name, "rb") as file:
            response = requests.post(
                "https://api.imgbb.com/1/upload",
                data={"key": api_key},
                files={"image": file}
            )
        return response.json()["data"]["url"]
    except Exception as e:
        return None

# Placeholders
video_placeholder = st.empty()
alert_placeholder = st.empty()
chart_placeholder = st.line_chart(pd.DataFrame({"Detections": []}))

# Session state
if "running" not in st.session_state:
    st.session_state.running = False
if "fire_log" not in st.session_state:
    st.session_state.fire_log = []
if "fire_active" not in st.session_state:
    st.session_state.fire_active = False
if "notification_sent" not in st.session_state:
    st.session_state.notification_sent = False
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = datetime.min

# Handle start/stop
if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# Stream loop
if st.session_state.running:
    cap = cv2.VideoCapture(0 if source == "Webcam" else r"C:\Users\nandish dave\OneDrive\Desktop\New folder\fire2.mp4")
    if not cap.isOpened():
        st.error("❌ Could not open video source.")
        st.stop()

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 20)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Video has ended or cannot be loaded.")
            break

        frame = cv2.resize(frame, (640, 480))
        results = model(frame, stream=True)

        fire_detected = False
        fire_count = 0

        for info in results:
            for box in info.boxes:
                confidence = int(box.conf[0] * 100)
                Class = int(box.cls[0])
                if Class == 0 and confidence > 50:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cvzone.putTextRect(frame, f"{classnames[Class]} {confidence}%", [x1+8, y1+30], scale=1, thickness=1)
                    fire_detected = True
                    fire_count += 1

        # Timestamp
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, now, (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        # Display
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Fire alert + WhatsApp notification
        if fire_detected:
            alert_placeholder.error("🚨 FIRE DETECTED! TAKE ACTION IMMEDIATELY 🚨")
            if not st.session_state.fire_active:
                st.session_state.fire_active = True

            # Send WhatsApp only every 30 seconds
            now_dt = datetime.now()
            if (now_dt - st.session_state.last_alert_time).total_seconds() > 30:
                img_url = upload_to_imgbb(frame, IMGBB_API_KEY)
                if img_url:
                    try:
                        client.messages.create(
                            from_=TWILIO_WHATSAPP_FROM,
                            to=TO_PHONE,
                            body=f"🔥 FIRE ALERT! Detected at {now_dt.strftime('%Y-%m-%d %H:%M:%S')}",
                            media_url=[img_url]
                        )
                        st.session_state.last_alert_time = now_dt
                        st.success("✅ WhatsApp alert sent with snapshot!")
                    except Exception as e:
                        st.error(f"Failed to send WhatsApp alert: {e}")
        else:
            alert_placeholder.empty()
            st.session_state.fire_active = False

        # Update fire log
        st.session_state.fire_log.append({"time": now, "detections": fire_count})
        df = pd.DataFrame(st.session_state.fire_log[-50:])
        chart_placeholder.line_chart(df.set_index("time"))

        time.sleep(1 / fps)

    cap.release()