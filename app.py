import streamlit as st
import cv2
import numpy as np
from ball_tracker import BallTracker


st.set_page_config(
    page_title="Cricket FairPlay",
    page_icon="🏏",
    layout="wide"
)


# -----------------------------
# PAGE TITLE
# -----------------------------

st.title("🏏 Cricket FairPlay")

st.caption(
    "Mobile camera based cricket ball tracking - V1"
)


# -----------------------------
# SESSION STATE
# -----------------------------

if "ball_count" not in st.session_state:
    st.session_state.ball_count = 0

if "over" not in st.session_state:
    st.session_state.over = 0

if "last_ball" not in st.session_state:
    st.session_state.last_ball = "Waiting..."

if "tracker" not in st.session_state:
    st.session_state.tracker = BallTracker()


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Match Control")

if st.sidebar.button("🔄 Reset Over"):

    st.session_state.ball_count = 0
    st.session_state.last_ball = "Waiting..."
    st.session_state.tracker = BallTracker()

    st.rerun()


# -----------------------------
# CAMERA
# -----------------------------

camera = st.camera_input(
    "📱 Take a frame from the mobile camera"
)


# -----------------------------
# MATCH INFORMATION
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Over",
        st.session_state.over
    )

with col2:
    st.metric(
        "Legal Balls",
        st.session_state.ball_count
    )

with col3:
    st.metric(
        "Last Ball",
        st.session_state.last_ball
    )


# -----------------------------
# PROCESS CAMERA IMAGE
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

        ball = st.session_state.tracker.update(frame)

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

            cv2.circle(
                display_frame,
                (x, y),
                3,
                (0, 0, 255),
                -1
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

            st.session_state.last_ball = "🟢 Ball detected"

        else:

            st.session_state.last_ball = "⚪ Ball not detected"

        display_frame = cv2.cvtColor(
            display_frame,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            display_frame,
            caption="Camera / Ball Detection",
            use_container_width=True
        )


# -----------------------------
# STATUS
# -----------------------------

st.divider()

st.subheader("Current Status")

if st.session_state.last_ball == "🟢 Ball detected":

    st.success("🏏 Cricket ball detected")

else:

    st.info(
        "Point the camera toward the cricket pitch."
    )


st.warning(
    "V1 is only for basic ball detection. "
    "Wide, No Ball and Wicket detection will be added later."
)
