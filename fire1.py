from ultralytics import YOLO
import cvzone
import cv2
import streamlit as st
import os
import time
from datetime import datetime
import pandas as pd
from twilio.rest import Client

st.set_page_config(page_title="Fire Detection System", layout="wide")
st.title("🔥 Real-Time Fire Detection Dashboard")

# Sidebar controls
source = st.sidebar.radio("Select Source", ["Webcam", "Sample Video"])
record_option = st.sidebar.checkbox("💾 Save Detection Video")
start_button = st.sidebar.button("▶ Start Stream")
stop_button = st.sidebar.button("⏹ Stop Stream")

# Load YOLO model
model = YOLO("fire.pt")
classnames = ["fire"]

# Twilio config
TWILIO_ACCOUNT_SID = "ACa74fab25d55343e1134cb6570a2ca890"
TWILIO_AUTH_TOKEN = "5e96bc18f67a3c2158276a14f6ea2f5e"
TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"  # Twilio sandbox number
TO_PHONE = "whatsapp:+918320437952"  # Your phone number

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Placeholders
video_placeholder = st.empty()
alert_placeholder = st.empty()
download_placeholder = st.sidebar.empty()
chart_placeholder = st.line_chart(pd.DataFrame({"Detections": []}))

# State variables
if "running" not in st.session_state:
    st.session_state.running = False
if "fire_log" not in st.session_state:
    st.session_state.fire_log = []
if "fire_active" not in st.session_state:
    st.session_state.fire_active = False
if "notification_sent" not in st.session_state:
    st.session_state.notification_sent = False

# Handle buttons
if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# Stream loop
if st.session_state.running:
    if source == "Webcam":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(r"C:\\Users\\nandish dave\\OneDrive\\Desktop\\New folder\\fire2.mp4")

    if not cap.isOpened():
        st.error("❌ Could not open video source.")
        st.stop()

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 20)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # VideoWriter setup
    out, save_path = None, None
    if record_option:
        os.makedirs("recordings", exist_ok=True)
        timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("recordings", f"fire_detection_{timestamp_filename}.mp4")
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        st.sidebar.success(f"Recording enabled → {save_path}")

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
                if Class == 0 and confidence > 50:  # Only fire class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cvzone.putTextRect(
                        frame,
                        f"{classnames[Class]} {confidence}%",
                        [x1 + 8, y1 + 30],
                        scale=1,
                        thickness=1,
                    )
                    fire_detected = True
                    fire_count += 1

        # Timestamp watermark
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, now, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Save frame
        if out:
            out.write(frame)

        # Display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        # Fire alert
        if fire_detected and not st.session_state.fire_active:
            alert_placeholder.error("🚨 FIRE DETECTED! TAKE ACTION IMMEDIATELY 🚨")
            st.session_state.fire_active = True

            # Play alert sound
            try:
                st.audio("alert.wav", autoplay=True)
            except Exception as e:
                st.warning(f"Could not play alert sound: {e}")

            # Send WhatsApp notification if not already sent
            if not st.session_state.notification_sent:
                try:
                    client.messages.create(
                        from_=TWILIO_WHATSAPP_FROM,
                        body=f"🔥 FIRE ALERT! Detected at {now}. Take immediate action!",
                        to=TO_PHONE
                    )
                    st.session_state.notification_sent = True
                    st.success("✅ WhatsApp alert sent!")
                except Exception as e:
                    st.error(f"Failed to send WhatsApp alert: {e}")
        elif not fire_detected:
            alert_placeholder.empty()
            st.session_state.fire_active = False
            st.session_state.notification_sent = False


        # Fire alert
        if fire_detected and not st.session_state.fire_active:
            alert_placeholder.error("🚨 FIRE DETECTED! TAKE ACTION IMMEDIATELY 🚨")
            st.session_state.fire_active = True

            # Send WhatsApp notification if not already sent
            if not st.session_state.notification_sent:
                try:
                    client.messages.create(
                        from_=TWILIO_WHATSAPP_FROM,
                        body=f"🔥 FIRE ALERT! Detected at {now}. Take immediate action!",
                        to=TO_PHONE
                    )
                    st.session_state.notification_sent = True
                    st.success("✅ WhatsApp alert sent!")
                except Exception as e:
                    st.error(f"Failed to send WhatsApp alert: {e}")
        elif not fire_detected:
            alert_placeholder.empty()
            st.session_state.fire_active = False
            st.session_state.notification_sent = False

        # Fire log update
        st.session_state.fire_log.append({"time": now, "detections": fire_count})
        df = pd.DataFrame(st.session_state.fire_log[-50:])
        chart_placeholder.line_chart(df.set_index("time"))

        # Small delay
        time.sleep(1 / fps)

    # Release resources
    cap.release()
    if out:
        out.release()

    # Add download button
    if save_path and os.path.exists(save_path):
        with open(save_path, "rb") as video_file:
            download_placeholder.download_button(
                label="⬇️ Download Recorded Video",
                data=video_file,
                file_name=os.path.basename(save_path),
                mime="video/mp4"
            )