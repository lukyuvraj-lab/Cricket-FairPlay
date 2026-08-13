import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Cricket FairPlay",
    page_icon="🏏",
    layout="centered"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🏏 Cricket FairPlay")
st.subheader("Green Tennis Cricket Ball Detection")

st.write(
    "Use your camera or upload an image to detect the cricket ball "
    "using the trained YOLO model."
)


# ---------------------------------------------------------
# MODEL PATH
# ---------------------------------------------------------

MODEL_PATH = Path("models/cricket_ball.pt")


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if MODEL_PATH.stat().st_size < 1000:
        raise ValueError(
            f"The model file appears to be empty or corrupted. "
            f"File size: {MODEL_PATH.stat().st_size} bytes."
        )

    return YOLO(str(MODEL_PATH))


try:
    model = load_model()

    st.success("✅ YOLO model loaded successfully")

except Exception as e:
    st.error("❌ Could not load the YOLO model")

    st.code(str(e))

    st.warning(
        "Make sure models/cricket_ball.pt is your real trained "
        "YOLO model file and is not empty or corrupted."
    )

    st.stop()


# ---------------------------------------------------------
# DETECTION SETTINGS
# ---------------------------------------------------------

st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.40,
    step=0.05
)

st.sidebar.write(
    f"Current confidence: **{confidence:.2f}**"
)


# ---------------------------------------------------------
# INPUT METHOD
# ---------------------------------------------------------

st.header("📷 Select Input")

input_method = st.radio(
    "Choose how you want to provide the image:",
    ["📷 Camera", "📁 Upload Image"],
    horizontal=True
)


image = None


# ---------------------------------------------------------
# CAMERA INPUT
# ---------------------------------------------------------

if input_method == "📷 Camera":

    camera_image = st.camera_input(
        "Take a photo of the cricket ball"
    )

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------

else:

    uploaded_file = st.file_uploader(
        "Upload a cricket-ball image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")


# ---------------------------------------------------------
# DETECTION
# ---------------------------------------------------------

if image is not None:

    st.divider()

    st.header("🔍 Detection Result")

    # Convert PIL image to NumPy
    image_array = np.array(image)

    # Display original image
    st.subheader("Original Image")

    st.image(
        image,
        caption="Input Image",
        use_container_width=True
    )

    # -----------------------------------------------------
    # RUN YOLO
    # -----------------------------------------------------

    with st.spinner("🔎 Detecting cricket ball..."):

        results = model.predict(
            source=image_array,
            conf=confidence,
            verbose=False
        )

    result = results[0]

    # -----------------------------------------------------
    # CHECK DETECTIONS
    # -----------------------------------------------------

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        st.error("❌ NO BALL DETECTED")

        st.info(
            "Try another image, reduce the confidence threshold, "
            "or make sure the cricket ball is clearly visible."
        )

    else:

        st.success("🏏 BALL DETECTED!")

        st.write(
            f"**Number of detections:** {len(boxes)}"
        )

        # -------------------------------------------------
        # DRAW BOUNDING BOXES
        # -------------------------------------------------

        annotated_image = result.plot()

        # YOLO returns BGR image.
        # Convert BGR → RGB.
        annotated_image = annotated_image[:, :, ::-1]

        st.subheader("🎯 Detected Ball")

        st.image(
            annotated_image,
            caption="YOLO Detection",
            use_container_width=True
        )

        # -------------------------------------------------
        # DETECTION DETAILS
        # -------------------------------------------------

        st.subheader("📊 Detection Details")

        for i, box in enumerate(boxes):

            confidence_score = float(box.conf[0])

            class_id = int(box.cls[0])

            # Get class name
            class_name = model.names.get(
                class_id,
                str(class_id)
            )

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            st.markdown(
                f"### 🏏 Detection {i + 1}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Confidence",
                    f"{confidence_score * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Class",
                    str(class_name)
                )

            st.write(
                f"**Bounding Box:** "
                f"X1={x1:.0f}, "
                f"Y1={y1:.0f}, "
                f"X2={x2:.0f}, "
                f"Y2={y2:.0f}"
            )

            st.divider()


# ---------------------------------------------------------
# INSTRUCTIONS
# ---------------------------------------------------------

else:

    st.info(
        "📷 Take a photo or 📁 upload an image to start detection."
    )

    st.markdown(
        """
        ### 💡 Tips for better detection

        - Keep the cricket ball clearly visible.
        - Use good lighting.
        - Avoid excessive motion blur.
        - Keep the ball reasonably large in the image.
        - Try lowering the confidence threshold if detection fails.
        """
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "🏏 Cricket FairPlay • YOLO Cricket Ball Detection"
)
