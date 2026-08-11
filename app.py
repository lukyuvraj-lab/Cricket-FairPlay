import streamlit as st
import cv2
import numpy as np


# -----------------------------
# Ball Tracker
# -----------------------------

class BallTracker:

    def __init__(self):
        self.previous_center = None
        self.ball_detected = False
        self.frames_without_ball = 0

    def detect_ball(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, 120])
        upper = np.array([180, 100, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)

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

        return best_circle

    def update(self, frame):

        ball = self.detect_ball(frame)

        if ball is not None:

            x, y, radius = ball

            self.previous_center = (x, y)

            self.ball_detected = True

            self.frames_without_ball = 0

            return ball

        self.frames_without_ball += 1

        if self.frames_without_ball > 10:

            self.ball_detected = False

        return None


# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(
    page_title="Cricket FairPlay",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 Cricket FairPlay")

st.caption(
    "Cricket ball tracking - V1"
)


# -----------------------------
# Session State
# -----------------------------

if "ball_count" not in st.session_state:
    st.session_state.ball_count = 0

if "last_ball" not in st.session_state:
    st.session_state.last_ball = "Waiting..."

if "tracker" not in st.session_state:
    st.session_state.tracker = BallTracker()


# -----------------------------
# Reset
# -----------------------------

if st.button("🔄 Reset"):

    st.session_state.ball_count = 0

    st.session_state.last_ball = "Waiting..."

    st.session_state.tracker = BallTracker()

    st.rerun()


# -----------------------------
# Camera
# -----------------------------

camera = st.camera_input(
    "📱 Take a picture using your mobile camera"
)


# -----------------------------
# Match Information
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Balls",
        st.session_state.ball_count
    )

with col2:

    st.metric(
        "Status",
        st.session_state.last_ball
    )


# -----------------------------
# Process Image
# -----------------------------

if camera is not None:

    bytes_data = camera.getvalue()

    image = np.asarray(
        bytearray(bytes_data),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image,
        cv2.IMREAD_COLOR
    )

    if frame is not None:

        ball = st.session_state.tracker.update(
            frame
        )

        display_frame = frame.copy()

        if ball is not None:

            x, y, radius = ball

            cv2.circle(
                display_frame,
                (x, y),
                radius,
                (0, 255, 0),
                3
            )

            cv2.putText(
                display_frame,
                "BALL",
                (x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            st.session_state.last_ball = (
                "🟢 Ball detected"
            )

        else:

            st.session_state.last_ball = (
                "⚪ Ball not detected"
            )

        display_frame = cv2.cvtColor(
            display_frame,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            display_frame,
            caption="Ball Detection",
            use_container_width=True
        )


st.divider()

st.info(
    "V1: Basic cricket ball detection. "
    "Wide, No Ball and Wicket detection "
    "will be added next."
)
