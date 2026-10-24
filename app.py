
import os
import cv2
import numpy as np
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="AI Safety Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom trained .pt model ka path environment variable se set ho sakta hai.
MODEL_PATH = os.getenv("YOLO_MODEL", "yolo11n.pt")

# Model file na ho to Ultralytics download karne ki koshish karega.
model = YOLO(MODEL_PATH)


@app.get("/")
def home():
    return {
        "status": "running",
        "message": "AI Safety Detection API is ready"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WEBP image."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum image size is 10 MB."
        )

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    results = model.predict(img, conf=0.25, verbose=False)
    predictions = []

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, confidence, class_id in zip(
            boxes, confidences, class_ids
        ):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(class_id)]

            predictions.append({
                "box": [x1, y1, x2, y2],
                "confidence": round(float(confidence), 4),
                "class": str(class_name)
            })

    return {"predictions": predictions}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
