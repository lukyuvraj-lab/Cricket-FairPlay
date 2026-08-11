import streamlit as st
from PIL import Image
import io

st.set_page_config(
    page_title="Cricket FairPlay",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 Cricket FairPlay")
st.write("Green Tennis Cricket Ball Detection")

# Load YOLO only after Streamlit starts
try:
    from ultralytics import YOLO
except Exception as e:
    st.error("YOLO/Ultralytics is not installed correctly.")
    st.code("python -m pip install ultralytics")
    st.exception(e)
    st.stop()

# Model
MODEL_PATH = "models/cricket_ball.pt"

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model: {MODEL_PATH}")
    st.exception(e)
    st.stop()

st.success("✅ Cricket ball model loaded")

st.subheader("📱 Camera")

photo = st.camera_input("Take a photo of the cricket ball")

if photo is not None:

    image_bytes = photo.getvalue()
    image = Image.open(io.BytesIO(image_bytes))

    st.image(
        image,
        caption="Camera image",
        use_container_width=True
    )

    st.subheader("🔍 Detection")

    results = model.predict(
        source=image,
        conf=0.40,
        imgsz=640
    )

    result_image = results[0].plot()

    st.image(
        result_image,
        caption="Detected Cricket Ball",
        use_container_width=True
    )

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:

        st.success(f"🏏 Cricket ball detected: {len(boxes)}")

        for box in boxes:
            confidence = float(box.conf[0])
            st.write(f"Confidence: **{confidence:.1%}**")

    else:
        st.warning("❌ Cricket ball not detected")
