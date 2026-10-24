import os
import cv2
import numpy as np
import uvicorn
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI(title="YOLOv11 Safety Detection API")

# Streamlit data transport compatibility enable karne ke liye settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "data.pkl"
BASE_MODEL = "yolo11n.pt"

# Pehle local baseline structure load system confirm karna
if os.path.exists(BASE_MODEL):
    model = YOLO(BASE_MODEL)
    print(f"🎉 Local base anchor weights successfully loaded from {BASE_MODEL}")
else:
    model = YOLO("yolo11n.pt")

# Custom Checkpoint validation load parameters dictionary matrix mapping
if os.path.exists(MODEL_PATH):
    try:
        print(f"⏳ Mapping custom binary dictionary structures from local path: {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                model.model.load_state_dict(checkpoint["model"].state_dict(), strict=False)
            elif "state_dict" in checkpoint:
                model.model.load_state_dict(checkpoint["state_dict"], strict=False)
            print("🎉 Custom weights successfully injected into YOLOv11 layers!")
        else:
            model = YOLO(MODEL_PATH)
            print("🎉 Custom weights loaded directly via binary object checkpoint mapping.")
    except Exception as e:
        print(f"⚠️ Safe exception catch - Layer serialization parameters skipped: {e}")
else:
    print("⚠️ data.pkl structure file missing context. Running backend via standard default configurations baseline mode.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image matrix structural layout"})
    
    results = model(img)
    predictions = []
    
    for result in results:
        if result.boxes is not None:
            boxes_np = result.boxes.xyxy.cpu().numpy()
            confs_np = result.boxes.conf.cpu().numpy()
            clss_np = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes_np)):
                x1, y1, x2, y2 = boxes_np[i]
                conf = float(confs_np[i])
                cls_id = int(clss_np[i])
                class_name = model.names[cls_id]
                
                predictions.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 2),
                    "class": class_name
                })
            
    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
