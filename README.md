# YOLOv11n Safety Detection Project (FastAPI + Streamlit)

This repository contains an end-to-end computer vision assignment project using **YOLOv11n** to detect helmets, license plates, and seatbelts. 

## 📁 Dataset Guide (What to do with your downloaded dataset)
Your downloaded dataset contains images and text labels. You need to use it to **train** your model so it learns what a helmet, license plate, or seatbelt looks like.
1. Put the dataset folder in your workspace.
2. Ensure it has a `data.yaml` file pointing to your train/val image paths.
3. Run the training command provided in the guide below to generate your custom `best.pt` file.

## 🚀 Setup & Installation

### Step 1: Install Dependencies
Run the following command in your terminal:
```bash
pip install fastapi uvicorn ultralytics streamlit opencv-python numpy python-multipart pillow requests
```

### Step 2: Project Structure
Organize your project folder like this:
```text
safety-detection-project/
│
├── app.py          # FastAPI Backend Server
├── streamlit_app.py # Streamlit Frontend Dashboard
├── train.py        # Python script to train your dataset
└── best.pt         # Your trained YOLOv11n weights (generated after training)
```

### Step 3: Run the Application
1. **Start the FastAPI Backend:**
   ```bash
   python app.py
   ```
   The backend will run on `http://127.0.0.1:8000`.

2. **Start the Streamlit Frontend:**
   Open a new terminal window and run:
   ```bash
   streamlit run streamlit_app.py
   ```
   The interactive dashboard will open automatically in your web browser!
