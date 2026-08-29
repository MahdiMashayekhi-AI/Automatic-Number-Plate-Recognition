import os
import cv2
import torch
import tempfile
import numpy as np
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, status
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from src.database.connection import get_db
from src.database.models import DetectedPlate
from src.pipeline import ANPRPipeline
from src.config import KEYPOINT_MODEL_PATH, OCR_MODEL_PATH


app = FastAPI(title="ANPR System API")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline = ANPRPipeline(KEYPOINT_MODEL_PATH, OCR_MODEL_PATH, DEVICE)


@app.get('/api/v1/status')
def get_service_status():
  return {"status": "online", "message": "ANPR Service is running"}


@app.get('/api/v1/plates/stats')
def get_plate_statistics(db: Session = Depends(get_db)):
   total = db.query(DetectedPlate).count()
   unique_plates = db.query(DetectedPlate.plate_text).distinct().count()
   avg_confidence = db.query(func.avg(DetectedPlate.confidence)).scalar()

   return {
      "total_detections": total,
      "unique_plates": unique_plates,
      "average_confidence": round(avg_confidence or 0, 2)
    }


@app.get('/api/v1/plates')
def get_plates(limit:int = Query(20, ge=1, le=100, description="Number of recoreds to return"), db:Session = Depends(get_db)):
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


@app.get('/api/v1/plates/search')
def search_plates(q:Optional[str] = Query(None, min_length=1, description="Plate number to search"), limit: int = Query(20, ge=1, le=100), db:Session = Depends(get_db)):
  query = db.query(DetectedPlate)

  if q:
     query = query.filter(DetectedPlate.plate_text.contains(q))

  plates = query.order_by(DetectedPlate.created_at.desc()).limit(limit).all()

  if not plates:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Np plates found matching: {q}")
  
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


@app.get('/api/v1/plates/{plate_id}')
def get_plate_by_id(plate_id: int, db: Session = Depends(get_db)):
  plate = db.query(DetectedPlate).filter(DetectedPlate.id == plate_id).first()

  if not plate:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plate with ID {plate_id} not found")

  return {
     "id": plate.id,
     "track_id": plate.track_id,
     "plate_text": plate.plate_text,
     "confidence": plate.confidence,
     "created_at": plate.created_at.isoformat()
  }


@app.post('/api/v1/plates/detect')
async def detect_plate_from_image(file: UploadFile = File(..., description="Image file containing vehicle plate")):
  if not file.content_type.startswith('image/'):
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be an image")


  contents = await file.read()
  nparr = np.frombuffer(contents, np.uint8)
  image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  if image is None:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file")

  return pipeline.process_image(image)


@app.post('/api/v1/plates/detect-from-video')
async def detect_plates_from_video(file: UploadFile = File(..., description="Video file for plate detection")):
    if not file.content_type.startswith('video/'):
       raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be a video")
    
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
                detail="Could not open video file"
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

    except HTTPException:
       raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not open video file")
    
    except Exception as e:
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing video: {str(e)}")

    finally:
        pipeline.reset()

        if 'cap' in locals():
            cap.release()

        tfile.close()

        if os.path.exists(temp_path):
            os.unlink(temp_path)