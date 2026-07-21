from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.models import DetectedPlate


app = FastAPI(
  title="ANPR System API"
)


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