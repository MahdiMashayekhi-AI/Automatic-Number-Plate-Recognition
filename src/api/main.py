import cv2
import torch
import tempfile
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.models import DetectedPlate
from src.pipeline import ANPRPipeline


app = FastAPI(
  title="ANPR System API"
)

TRACKER_MODEL = "license_plate_keypoint.pt"
OCR_MODEL = "outputs/checkpoints/best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline = ANPRPipeline(TRACKER_MODEL, OCR_MODEL, DEVICE)


@app.get('/')
def read_root():
  return {"status": "online", "message": "ANPR Service is running"}


@app.get('/plates')
def get_plates(limit:int = 20, db:Session = Depends(get_db)):
  plates = db.query(DetectedPlate).order_by(DetectedPlate.created_at.desc()).limit(limit).all()
  return [
    {
      "id": plate.id,
      "track_id": plate.track_id,
      "plate_text": plate.plate_text,
      "confidence": plate.confidence,
      "created_at": plate.created_at
    }
    for plate in plates
  ]


@app.get('/plates/search')
def plate_search(q:str = Query(..., description="Plate Number"), db:Session = Depends(get_db)):
  plates = db.query(DetectedPlate).filter(DetectedPlate.plate_text.contains(q)).all()

  if not plates:
    raise HTTPException(status_code=404, detail=f"Number plates with {q} not found!")
  
  return [
    {
      "id": plate.id,
      "track_id": plate.track_id,
      "plate_text": plate.plate_text,
      "confidence": plate.confidence,
      "created_at": plate.created_at
    }
    for plate in plates
  ]


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
  contents = await file.read()
  nparr = np.frombuffer(contents, np.uint8)
  image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  if image is None:
    raise HTTPException(status_code=400, detail="File is not a valid image!")

  return pipeline.process_image(image)


@app.post('/predict/video')
async def predict_video(file: UploadFile = File(...)):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = tfile.name

    try:
        contents = await file.read()
        tfile.write(contents)
        tfile.close()

        pipeline.reset()

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open video file!"
            )

        results = {}

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_results = pipeline.process_frame(frame)

            for track_id, state in frame_results.items():
                current_text = state.get("text")

                if track_id not in results:
                    results[track_id] = state
                else:
                    previous_text = results[track_id].get("text")

                    if (current_text != "Detecting" or previous_text == "Detecting"):
                        results[track_id] = state

        cap.release()

        return [
            {
                "track_id": track_id,
                "plate_text": state["text"],
                "conf": state["conf"]
            }
            for track_id, state in results.items()
            if state.get("text") != "Detecting"
        ]

    finally:
        pipeline.reset()

        if 'cap' in locals():
            cap.release()

        tfile.close()

        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)