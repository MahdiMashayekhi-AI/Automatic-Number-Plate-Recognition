from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from src.database.connection import Base


class DetectedPlate(Base):
  __tablename__ = "detected_plates"

  id = Column(Integer, primary_key=True, index=True, autoincrement=True)
  track_id = Column(Integer, index=True)
  plate_text = Column(String(8), nullable=False)
  confidence = Column(Float, nullable=False)
  created_at = Column(DateTime, default=datetime.now)