from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import shutil
import os
import cv2
import traceback

from app.model import model
import os

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

static_dir = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

UPLOAD_FOLDER = os.path.join(static_dir, "uploads")
OUTPUT_FOLDER = os.path.join(static_dir, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(" App started successfully")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    try:
        print("\n===== NEW REQUEST =====")

        # Save uploaded file
        filename = os.path.basename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(" File saved at:", file_path)

        # Run model
        results = model.predict(source=file_path, conf=0.25, save=False)

        r = results[0]

        print(" Total detections:", len(r.boxes))

        detections = []

        if len(r.boxes) == 0:
            print(" No detections found")
        else:
            for i in range(len(r.boxes)):

                cls = int(r.boxes.cls[i])
                conf = float(r.boxes.conf[i])

                # Bounding box (x1, y1, x2, y2)
                xyxy = r.boxes.xyxy[i].tolist()

                print(f"Detection {i}:")
                print("   Class:", cls)
                print("   Confidence:", conf)
                print("   Box:", xyxy)

                detections.append({
                    "class": cls,
                    "confidence": round(conf, 3),
                    "box": [round(x, 2) for x in xyxy]
                })

        # Generate annotated image
        annotated_frame = r.plot()

        output_path = os.path.join(OUTPUT_FOLDER, filename)

        cv2.imwrite(output_path, annotated_frame)

        print(" Output image saved:", output_path)

        # Return to frontend
        return templates.TemplateResponse("index.html", {
            "request": request,
            "output_image": f"/static/outputs/{filename}",
            "detections": detections
        })

    except Exception as e:
        print(" ERROR:", str(e))
        print(traceback.format_exc())

        return JSONResponse(content={"error": str(e)})
