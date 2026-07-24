import cv2
import torch
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
OCR_MODEL = "outputs/best_model.pt"
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
  frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  if frame is None:
    raise HTTPException(status_code=400, detail="File is not a valid image!")

  results = pipeline.process_frame(frame)

  return [
    {
      "track_id": track_id,
      "plate_text": data['text'],
      "confidence": data['conf'],
      "bbox": data['bbox']
    }
    for track_id, data in results.items()
  ]