
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="AI Safety Detection Dashboard",
    page_icon="🪖",
    layout="wide"
)

st.title("🪖 AI Safety Detection Dashboard")
st.write("Upload an image to detect objects using YOLO.")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        if st.button("Run AI Detection 🚀"):
            with st.spinner("Loading YOLO and detecting objects..."):
                model = load_model()

                image_array = np.array(image)

                results = model.predict(
                    source=image_array,
                    conf=conf_threshold,
                    verbose=False
                )

                # Ultralytics plot returns a BGR image
                annotated_image = results[0].plot()
                annotated_image = annotated_image[:, :, ::-1]

                with col2:
                    st.subheader("Detection Result")
                    st.image(
                        annotated_image,
                        use_container_width=True
                    )

                boxes = results[0].boxes

                count = len(boxes) if boxes is not None else 0

                st.success(f"Detection complete! Objects found: {count}")

                if boxes is not None and count > 0:
                    st.subheader("Detection Details")

                    details = []

                    for box in boxes:
                        class_id = int(box.cls.item())
                        confidence = float(box.conf.item())

                        details.append({
                            "Object": model.names[class_id],
                            "Confidence": f"{confidence:.1%}"
                        })

                    st.dataframe(details, use_container_width=True)

                if count == 0:
                    st.info("No objects detected above this confidence threshold.")

    except Exception as e:
        st.error(f"Detection failed: {e}")
