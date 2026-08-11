import streamlit as st
import cv2
import av

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Cricket FairPlay",
    page_icon="🏏",
    layout="wide"
)


# ==========================================
# LOAD TRAINED YOLO MODEL
# ==========================================

MODEL_PATH = "models/cricket_ball.pt"

model = YOLO(MODEL_PATH)


# ==========================================
# LIVE VIDEO PROCESSOR
# ==========================================

class CricketVideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        # ----------------------------------
        # YOLO BALL DETECTION
        # ----------------------------------

        results = model.predict(
            image,
            conf=0.40,
            imgsz=640,
            verbose=False
        )

        # ----------------------------------
        # DRAW DETECTIONS
        # ----------------------------------

        for result in results:

            boxes = result.boxes

            for box in boxes:

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Draw box
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                # Label
                label = f"CRICKET BALL {confidence:.2f}"

                cv2.putText(
                    image,
                    label,
                    (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                cv2.circle(
                    image,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1
                )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ==========================================
# UI
# ==========================================

st.title("🏏 Cricket FairPlay")

st.subheader(
    "V3 — AI Green Ball Detection"
)

st.write(
    "Point your mobile camera at the green tennis ball."
)


# ==========================================
# LIVE CAMERA
# ==========================================

webrtc_ctx = webrtc_streamer(
    key="cricket-fairplay-camera",

    video_processor_factory=CricketVideoProcessor,

    media_stream_constraints={
        "video": {
            "facingMode": "environment"
        },
        "audio": False
    },

    async_processing=True
)


# ==========================================
# STATUS
# ==========================================

st.divider()

if webrtc_ctx.state.playing:

    st.success(
        "🟢 Live camera is running"
    )

else:

    st.info(
        "📱 Press START to start the camera"
    )


st.caption(
    "AI model: Cricket FairPlay Green Tennis Ball"
)
