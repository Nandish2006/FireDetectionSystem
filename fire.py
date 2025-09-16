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




























# from ultralytics import YOLO
# import cvzone
# import cv2
# import streamlit as st
# import os
# import time
# from datetime import datetime
# import pandas as pd
# from twilio.rest import Client

# st.set_page_config(page_title="Fire Detection System", layout="wide")
# st.title("🔥 Real-Time Fire Detection Dashboard")

# # Sidebar controls
# source = st.sidebar.radio("Select Source", ["Webcam", "Sample Video"])
# record_option = st.sidebar.checkbox("💾 Save Detection Video")
# start_button = st.sidebar.button("▶ Start Stream")
# stop_button = st.sidebar.button("⏹ Stop Stream")

# # Load YOLO model once
# if "model" not in st.session_state:
#     st.session_state.model = YOLO("fire.pt")
# classnames = ["fire"]

# # Twilio config
# TWILIO_ACCOUNT_SID = "xxxx"
# TWILIO_AUTH_TOKEN = "xxxx"
# TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"
# TO_PHONE = "whatsapp:+918320437952"

# client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# # Placeholders
# video_placeholder = st.empty()
# alert_placeholder = st.empty()
# download_placeholder = st.sidebar.empty()
# chart_placeholder = st.line_chart(pd.DataFrame({"Detections": []}))

# # State variables
# if "running" not in st.session_state:
#     st.session_state.running = False
# if "fire_log" not in st.session_state:
#     st.session_state.fire_log = []
# if "fire_active" not in st.session_state:
#     st.session_state.fire_active = False
# if "notification_sent" not in st.session_state:
#     st.session_state.notification_sent = False
# if "writer" not in st.session_state:
#     st.session_state.writer = None
# if "save_path" not in st.session_state:
#     st.session_state.save_path = None

# # Handle buttons
# if start_button:
#     st.session_state.running = True
# if stop_button:
#     st.session_state.running = False
#     if "cap" in st.session_state and st.session_state.cap:
#         st.session_state.cap.release()
#     if st.session_state.writer:
#         st.session_state.writer.release()
#     if st.session_state.save_path and os.path.exists(st.session_state.save_path):
#         with open(st.session_state.save_path, "rb") as video_file:
#             download_placeholder.download_button(
#                 label="⬇️ Download Recorded Video",
#                 data=video_file,
#                 file_name=os.path.basename(st.session_state.save_path),
#                 mime="video/mp4"
#             )

# # Stream loop
# if st.session_state.running:
#     # Open capture only once
#     if "cap" not in st.session_state or st.session_state.cap is None or not st.session_state.cap.isOpened():
#         if source == "Webcam":
#             st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#         else:
#             st.session_state.cap = cv2.VideoCapture(r"C:\\Users\\nandish dave\\OneDrive\\Desktop\\New folder\\fire2.mp4")

#         if not st.session_state.cap.isOpened():
#             st.error("❌ Could not open video source.")
#             st.stop()

#         fps = int(st.session_state.cap.get(cv2.CAP_PROP_FPS) or 20)
#         width = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
#         height = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

#         if record_option:
#             os.makedirs("recordings", exist_ok=True)
#             timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
#             st.session_state.save_path = os.path.join("recordings", f"fire_detection_{timestamp_filename}.mp4")
#             st.session_state.writer = cv2.VideoWriter(
#                 st.session_state.save_path,
#                 cv2.VideoWriter_fourcc(*'mp4v'),
#                 fps,
#                 (width, height)
#             )
#             st.sidebar.success(f"Recording enabled → {st.session_state.save_path}")

#     cap = st.session_state.cap
#     model = st.session_state.model

#     ret, frame = cap.read()
#     if not ret:
#         st.warning("⚠️ Video has ended or cannot be loaded.")
#         st.session_state.running = False
#     else:
#         frame = cv2.resize(frame, (640, 480))
#         results = model(frame, stream=True)

#         fire_detected = False
#         fire_count = 0

#         for info in results:
#             for box in info.boxes:
#                 confidence = int(box.conf[0] * 100)
#                 Class = int(box.cls[0])
#                 if Class == 0 and confidence > 50:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
#                     cvzone.putTextRect(
#                         frame,
#                         f"{classnames[Class]} {confidence}%",
#                         [x1 + 8, y1 + 30],
#                         scale=1,
#                         thickness=1,
#                     )
#                     fire_detected = True
#                     fire_count += 1

#         now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         cv2.putText(frame, now, (10, frame.shape[0] - 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         if st.session_state.writer:
#             st.session_state.writer.write(frame)

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         video_placeholder.image(frame_rgb, channels="RGB")

#         if fire_detected and not st.session_state.fire_active:
#             alert_placeholder.error("🚨 FIRE DETECTED! TAKE ACTION IMMEDIATELY 🚨")
#             st.session_state.fire_active = True
#             if not st.session_state.notification_sent:
#                 try:
#                     client.messages.create(
#                         from_=TWILIO_WHATSAPP_FROM,
#                         body=f"🔥 FIRE ALERT! Detected at {now}. Take immediate action!",
#                         to=TO_PHONE
#                     )
#                     st.session_state.notification_sent = True
#                     st.success("✅ WhatsApp alert sent!")
#                 except Exception as e:
#                     st.error(f"Failed to send WhatsApp alert: {e}")
#         elif not fire_detected:
#             alert_placeholder.empty()
#             st.session_state.fire_active = False
#             st.session_state.notification_sent = False

#         st.session_state.fire_log.append({"time": now, "detections": fire_count})
#         df = pd.DataFrame(st.session_state.fire_log[-50:])
#         chart_placeholder.line_chart(df.set_index("time"))

























# from ultralytics import YOLO
# import cvzone
# import cv2
# import streamlit as st
# import os
# import time
# from datetime import datetime
# import pandas as pd
# from twilio.rest import Client

# st.set_page_config(page_title="Fire Detection System", layout="wide")
# st.title("🔥 Real-Time Fire Detection Dashboard")

# # Twilio WhatsApp config
# TWILIO_ACCOUNT_SID = "YOUR_ACCOUNT_SID"
# TWILIO_AUTH_TOKEN = "YOUR_AUTH_TOKEN"
# WHATSAPP_FROM = "whatsapp:+14155238886"  # Twilio Sandbox number
# WHATSAPP_TO = "whatsapp:+91832XXXXXXX"   # Your phone number

# client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# def send_whatsapp_alert(message):
#     try:
#         client.messages.create(
#             body=message,
#             from_=WHATSAPP_FROM,
#             to=WHATSAPP_TO
#         )
#     except Exception as e:
#         st.error(f"WhatsApp notification failed: {e}")

# # Sidebar controls
# source = st.sidebar.radio("Select Source", ["Webcam", "Sample Video"])
# record_option = st.sidebar.checkbox("💾 Save Detection Video")
# start_button = st.sidebar.button("▶ Start Stream")
# stop_button = st.sidebar.button("⏹ Stop Stream")

# # Load YOLO model
# model = YOLO("yolov8n.pt")  # replace with your trained fire model if available
# classnames = ["fire"]

# # Placeholders
# video_placeholder = st.empty()
# alert_placeholder = st.empty()
# download_placeholder = st.sidebar.empty()
# chart_placeholder = st.line_chart(pd.DataFrame({"Detections": []}))

# # Session state
# if "running" not in st.session_state:
#     st.session_state.running = False
# if "fire_log" not in st.session_state:
#     st.session_state.fire_log = []
# if "fire_alert_sent" not in st.session_state:
#     st.session_state.fire_alert_sent = False  # To avoid spamming messages

# # Handle buttons
# if start_button:
#     st.session_state.running = True
# if stop_button:
#     st.session_state.running = False

# cap = None
# out = None
# save_path = None

# try:
#     if st.session_state.running:
#         if source == "Webcam":
#             cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#         else:
#             cap = cv2.VideoCapture(r"C:\Users\nandish dave\OneDrive\Desktop\New folder\fire2.mp4")

#         if not cap.isOpened():
#             st.error("❌ Could not open video source.")
#             st.stop()

#         fps = int(cap.get(cv2.CAP_PROP_FPS) or 20)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

#         if record_option:
#             os.makedirs("recordings", exist_ok=True)
#             timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
#             save_path = os.path.join("recordings", f"fire_detection_{timestamp_filename}.mp4")
#             out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
#             st.sidebar.success(f"Recording enabled → {save_path}")

#         fire_active = False

#         while st.session_state.running:
#             ret, frame = cap.read()
#             if not ret:
#                 st.warning("⚠️ Video has ended or cannot be loaded.")
#                 break

#             frame = cv2.resize(frame, (640, 480))
#             results = model(frame)

#             fire_detected = False
#             fire_count = 0

#             for info in results:
#                 if hasattr(info, 'boxes'):
#                     for box in info.boxes:
#                         conf = int(box.conf[0] * 100)
#                         cls = int(box.cls[0])
#                         if conf > 50 and cls < len(classnames) and classnames[cls] == "fire":
#                             x1, y1, x2, y2 = map(int, box.xyxy[0])
#                             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
#                             cvzone.putTextRect(frame, f"{classnames[cls]} {conf}%", [x1+8, y1+30], scale=1, thickness=1)
#                             fire_detected = True
#                             fire_count += 1

#             now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             cv2.putText(frame, now, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

#             if out:
#                 out.write(frame)

#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             video_placeholder.image(frame_rgb, channels="RGB")

#             # Fire alert banner
#             if fire_detected and not fire_active:
#                 alert_placeholder.error("🚨 FIRE DETECTED! TAKE ACTION IMMEDIATELY 🚨")
#                 fire_active = True

#                 # Send WhatsApp notification once per event
#                 if not st.session_state.fire_alert_sent:
#                     send_whatsapp_alert(f"🚨 FIRE ALERT at {now}! Check immediately.")
#                     st.session_state.fire_alert_sent = True

#             elif not fire_detected:
#                 alert_placeholder.empty()
#                 fire_active = False
#                 st.session_state.fire_alert_sent = False  # reset to send next alert

#             st.session_state.fire_log.append({"time": now, "detections": fire_count})
#             df = pd.DataFrame(st.session_state.fire_log[-50:])
#             chart_placeholder.line_chart(df.set_index("time"))

#             time.sleep(1 / fps)

# finally:
#     if cap and cap.isOpened():
#         cap.release()
#     if out:
#         out.release()
#     if save_path and os.path.exists(save_path):
#         with open(save_path, "rb") as video_file:
#             download_placeholder.download_button(
#                 label="⬇️ Download Recorded Video",
#                 data=video_file,
#                 file_name=os.path.basename(save_path),
#                 mime="video/mp4"
#             )




















# from ultralytics import YOLO
# import cvzone
# import cv2
# import streamlit as st
# import os
# import time
# from datetime import datetime
# import pandas as pd

# st.set_page_config(page_title="Fire Detection System", layout="wide")
# st.title("🔥 Real-Time Fire Detection Dashboard")

# # Sidebar controls
# source = st.sidebar.radio("Select Source", ["Webcam", "Sample Video"])
# record_option = st.sidebar.checkbox("💾 Save Detection Video")
# start_button = st.sidebar.button("▶ Start Stream")
# stop_button = st.sidebar.button("⏹ Stop Stream")

# # Load YOLO model
# model = YOLO("fire.pt")
# classnames = ["fire"]

# # Placeholders
# video_placeholder = st.empty()
# alert_placeholder = st.empty()
# download_placeholder = st.sidebar.empty()
# chart_placeholder = st.line_chart(pd.DataFrame({"Detections": []}))

# # State variables
# if "running" not in st.session_state:
#     st.session_state.running = False
# if "fire_log" not in st.session_state:
#     st.session_state.fire_log = []

# # Handle buttons
# if start_button:
#     st.session_state.running = True
# if stop_button:
#     st.session_state.running = False

# # Stream loop
# if st.session_state.running:
#     if source == "Webcam":
#         cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     else:
#         cap = cv2.VideoCapture(r"C:\\Users\\nandish dave\\OneDrive\\Desktop\\New folder\\fire2.mp4")

#     if not cap.isOpened():
#         st.error("❌ Could not open video source.")
#         st.stop()

#     # Video properties
#     fps = int(cap.get(cv2.CAP_PROP_FPS) or 20)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

#     # VideoWriter setup
#     out, save_path = None, None
#     if record_option:
#         os.makedirs("recordings", exist_ok=True)
#         timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
#         save_path = os.path.join("recordings", f"fire_detection_{timestamp_filename}.mp4")
#         out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
#         st.sidebar.success(f"Recording enabled → {save_path}")

#     fire_active = False

#     while st.session_state.running:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("⚠️ Video has ended or cannot be loaded.")
#             break

#         frame = cv2.resize(frame, (640, 480))
#         results = model(frame, stream=True)

#         fire_detected = False
#         fire_count = 0

#         for info in results:
#             for box in info.boxes:
#                 confidence = int(box.conf[0] * 100)
#                 Class = int(box.cls[0])
#                 if confidence > 50:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
#                     cvzone.putTextRect(
#                         frame,
#                         f"{classnames[Class]} {confidence}%",
#                         [x1 + 8, y1 + 30],
#                         scale=1,
#                         thickness=1,
#                     )
#                     fire_detected = True
#                     fire_count += 1

#         # Timestamp watermark
#         now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         cv2.putText(frame, now, (10, height - 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

#         # Save frame
#         if out:
#             out.write(frame)

#         # Display in Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         video_placeholder.image(frame_rgb, channels="RGB")

#         # Fire alert
#         if fire_detected and not fire_active:
#             alert_placeholder.error("🚨 FIRE DETECTED! TAKE ACTION IMMEDIATELY 🚨")
#             fire_active = True
#         elif not fire_detected:
#             alert_placeholder.empty()
#             fire_active = False

#         # Fire log update
#         st.session_state.fire_log.append({"time": now, "detections": fire_count})
#         df = pd.DataFrame(st.session_state.fire_log[-50:])
#         chart_placeholder.line_chart(df.set_index("time"))

#         # Small delay for smoothness
#         time.sleep(1 / fps)

#     # Release resources
#     cap.release()
#     if out:
#         out.release()

#     # ✅ Add download button after release
#     if save_path and os.path.exists(save_path):
#         with open(save_path, "rb") as video_file:
#             download_placeholder.download_button(
#                 label="⬇️ Download Recorded Video",
#                 data=video_file,
#                 file_name=os.path.basename(save_path),
#                 mime="video/mp4"
#             )














































# import cv2
# import math
# import streamlit as st
# from ultralytics import YOLO
# import cvzone
# from datetime import datetime
# import os

# # Load YOLO model
# model = YOLO("fire.pt")

# # Streamlit UI
# st.title("🔥 Fire Detection System (YOLOv8 + Webcam)")
# st.markdown("Live detection from webcam feed.")

# # Session state
# if "running" not in st.session_state:
#     st.session_state.running = False
# if "fire_logs" not in st.session_state:
#     st.session_state.fire_logs = []  # store history of fire detections

# # Status indicator
# status_placeholder = st.empty()
# time_placeholder = st.empty()
# log_placeholder = st.container()

# if st.session_state.running:
#     status_placeholder.markdown("🟢 **Status: Running**")
# else:
#     status_placeholder.markdown("🔴 **Status: Stopped**")

# # Buttons
# col1, col2, col3 = st.columns(3)
# with col1:
#     if st.button("▶️ Start Detection"):
#         st.session_state.running = True
# with col2:
#     if st.button("⏹️ Stop Detection"):
#         st.session_state.running = False
# with col3:
#     if st.button("🧹 Clear Logs"):
#         st.session_state.fire_logs = []
#         open("fire_logs.txt", "w").close()  # clear file too

# # Update status
# if st.session_state.running:
#     status_placeholder.markdown("🟢 **Status: Running**")
# else:
#     status_placeholder.markdown("🔴 **Status: Stopped**")

# # Placeholders
# frame_placeholder = st.empty()
# sound_placeholder = st.empty()

# # Webcam capture
# cap = cv2.VideoCapture(0)

# classnames = ['fire']
# ALERT_SOUND = "alert.wav"

# # Ensure log file exists
# if not os.path.exists("fire_logs.txt"):
#     open("fire_logs.txt", "w").close()

# # Main loop
# while st.session_state.running and cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         st.warning("⚠️ Could not access webcam. Please check your camera.")
#         break

#     frame = cv2.resize(frame, (640, 480))
#     results = model(frame, stream=True)

#     fire_detected = False

#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             confidence = math.ceil(box.conf[0] * 100)
#             Class = int(box.cls[0])
#             if confidence > 50:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
#                 cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%',
#                                    [x1 + 8, y1 + 30],
#                                    scale=1, thickness=1)
#                 fire_detected = True

#     # Show frame
#     frame_placeholder.image(frame, channels="BGR")

#     # Play sound + log if fire detected
#     if fire_detected:
#         sound_placeholder.audio(ALERT_SOUND, autoplay=True)
#         detection_time = datetime.now().strftime("%H:%M:%S")
#         log_entry = f"🔥 Fire detected at {detection_time}"
#         st.session_state.fire_logs.append(log_entry)

#         # Save to file
#         with open("fire_logs.txt", "a") as f:
#             f.write(log_entry + "\n")
#     else:
#         sound_placeholder.empty()


#     # Show last checked time
#     current_time = datetime.now().strftime("%H:%M:%S")
#     time_placeholder.markdown(f"🕒 **Last checked at:** {current_time}")

#     import time
#     import streamlit as st
#     import os

#     # --- Fire detection log writing (with emoji support) ---
#     log_entry = f"🔥 Fire detected at {time.strftime('%Y-%m-%d %H:%M:%S')}"
#     with open("fire_log.txt", "a", encoding="utf-8") as f:
#         f.write(log_entry + "\n")

#     # --- Fire log controls ---
#     col1, col2 = st.columns(2)

#     with col1:
#         if st.button("📖 View Fire Log"):
#             try:
#                 with open("fire_log.txt", "r", encoding="utf-8") as f:
#                     log_data = f.read()
#                 st.text_area("🔥 Fire Detection Log", log_data, height=200)
#             except FileNotFoundError:
#                 st.warning("⚠️ No fire log found yet!")

#     with col2:
#         if st.button("🧹 Clear Log"):
#             if os.path.exists("fire_log.txt"):
#                 open("fire_log.txt", "w", encoding="utf-8").close()  # clear file
#                 st.success("✅ Fire log cleared!")
#             else:
#                 st.warning("⚠️ No fire log to clear!")


    # # Show logs
    # with log_placeholder:
    #     st.subheader("📜 Fire Detection Logs")
    #     if st.session_state.fire_logs:
    #         for log in st.session_state.fire_logs[-10:][::-1]:
    #             st.markdown(log)
    #     else:
    #         st.markdown("✅ No logs available.")

# Cleanup
# cap.release()












































# from ultralytics import YOLO
# import cvzone
# import cv2
# import math




# # Running real time from webcam
# cap = cv2.VideoCapture('fire2.mp4')
# model = YOLO('fire.pt')


# # Reading the classes
# classnames = ['fire']

# while True:
#     ret,frame = cap.read()
#     frame = cv2.resize(frame,(640,480))
#     result = model(frame,stream=True)

#     # Getting bbox,confidence and class names informations to work with
#     for info in result:
#         boxes = info.boxes
#         for box in boxes:
#             confidence = box.conf[0]
#             confidence = math.ceil(confidence * 100)
#             Class = int(box.cls[0])
#             if confidence > 50:
#                 x1,y1,x2,y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
#                 cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
#                 cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
#                                    scale=1.5,thickness=2)


# import streamlit as st
# st.image(frame, channels="BGR")


# import matplotlib.pyplot as plt

# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()