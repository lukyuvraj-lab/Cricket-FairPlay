import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Cricket FairPlay",
    page_icon="🏏",
    layout="wide"
)


# ==========================================
# VIDEO PROCESSOR
# ==========================================

class CricketVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.ball_detected = False
        self.ball_x = 0
        self.ball_y = 0
        self.ball_radius = 0

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        # ----------------------------------
        # TEMPORARY BALL DETECTION
        # ----------------------------------
        # This is NOT the final AI detector.
        # We will replace this with a trained
        # cricket-ball model in V3.

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        lower = np.array([0, 0, 120])
        upper = np.array([180, 100, 255])

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_circle = None
        best_score = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 30 or area > 5000:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter == 0:
                continue

            circularity = (
                4 * np.pi * area
                / (perimeter * perimeter)
            )

            if circularity < 0.45:
                continue

            (x, y), radius = cv2.minEnclosingCircle(
                contour
            )

            if radius < 3 or radius > 50:
                continue

            score = circularity * area

            if score > best_score:

                best_score = score

                best_circle = (
                    int(x),
                    int(y),
                    int(radius)
                )

        # ----------------------------------
        # DRAW DETECTION
        # ----------------------------------

        if best_circle is not None:

            x, y, radius = best_circle

            self.ball_detected = True
            self.ball_x = x
            self.ball_y = y
            self.ball_radius = radius

            cv2.circle(
                image,
                (x, y),
                radius,
                (0, 255, 0),
                3
            )

            cv2.circle(
                image,
                (x, y),
                4,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                image,
                "BALL?",
                (x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            self.ball_detected = False

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ==========================================
# TITLE
# ==========================================

st.title("🏏 Cricket FairPlay")

st.subheader(
    "V2 - Live Mobile Camera"
)

st.write(
    "Live camera for future cricket-ball tracking."
)


# ==========================================
# STATUS
# ==========================================

status_placeholder = st.empty()


# ==========================================
# LIVE CAMERA
# ==========================================

webrtc_ctx = webrtc_streamer(
    key="cricket-camera",
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
# INFORMATION
# ==========================================

st.divider()

st.subheader("📊 Camera Status")

if webrtc_ctx.state.playing:

    st.success(
        "🟢 Live camera is running"
    )

else:

    st.info(
        "📱 Press START to activate the camera"
    )


st.warning(
    "V2 uses a temporary detector. "
    "It may detect other bright objects. "
    "The proper cricket-ball AI detector "
    "will be added in V3."
)
