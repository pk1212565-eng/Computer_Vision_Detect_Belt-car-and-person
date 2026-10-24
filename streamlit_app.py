
import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="AI Safety Detection Dashboard",
    layout="wide"
)

st.title("AI Safety Violation Detection Dashboard")
st.write("Detect objects using your YOLO backend API.")

# Local development ke liye:
API_URL = "http://127.0.0.1:8000/predict"

st.sidebar.header("Configuration")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

COLOR_MAP = {
    "helmet": (0, 255, 0),
    "license plate": (0, 255, 255),
    "seatbelt": (255, 0, 0),
    "person": (255, 128, 0),
    "car": (128, 0, 128)
}

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        st.error("Image open nahi ho saki. Doosri image try karo.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Result")

        if st.button("Run AI Detection"):
            try:
                with st.spinner("AI image process kar raha hai..."):
                    files = {
                        "file": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type or "image/jpeg"
                        )
                    }

                    response = requests.post(
                        API_URL,
                        files=files,
                        timeout=120
                    )

                    response.raise_for_status()
                    data = response.json()
                    predictions = data.get("predictions", [])

                    img_cv = cv2.cvtColor(
                        np.array(image),
                        cv2.COLOR_RGB2BGR
                    )

                    detections_count = 0

                    for pred in predictions:
                        confidence = float(pred["confidence"])

                        if confidence < conf_threshold:
                            continue

                        x1, y1, x2, y2 = map(
                            int, pred["box"]
                        )

                        class_name = str(pred["class"])
                        label = (
                            f"{class_name} "
                            f"{confidence * 100:.1f}%"
                        )

                        color = COLOR_MAP.get(
                            class_name.lower(),
                            (255, 255, 255)
                        )

                        cv2.rectangle(
                            img_cv,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )

                        cv2.putText(
                            img_cv,
                            label,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                        detections_count += 1

                    img_result = cv2.cvtColor(
                        img_cv,
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        img_result,
                        use_container_width=True
                    )

                    st.success(
                        f"Detected objects: {detections_count}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "FastAPI backend connect nahi hua. "
                    "Backend URL aur server check karo."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "Backend response mein der ho rahi hai. "
                    "Dobara try karo."
                )

            except requests.exceptions.HTTPError as e:
                st.error(f"Backend API error: {e}")

            except (KeyError, ValueError, TypeError) as e:
                st.error(
                    f"Prediction response ka format ghalat hai: {e}"
                )

            except Exception as e:
                st.error(f"Unexpected error: {e}")
