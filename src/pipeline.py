import cv2
from src.tracking.tracker import PlateTracker
from src.ocr.predictor import PlateReader
from src.decision.voting import TemporalVoter
from src.database.connection import get_db_context, Base, engine
from src.database.models import DetectedPlate


class ANPRPipeline:
  def __init__(self, tracker_model_path, ocr_model_path, device=None, blur_threshold=70):
    Base.metadata.create_all(bind=engine)

    self.tracker = PlateTracker(tracker_model_path)
    self.reader = PlateReader(ocr_model_path, device)
    self.voter = TemporalVoter()

    self.missing_frames = {}
    self.saved_track = set()
    self.blur_threshold = blur_threshold

  def process_frame(self, frame):
    frame_results = {}
    
    current_frame_outputs = self.tracker.track(frame)
    for track_id, state in current_frame_outputs.items():
      self.missing_frames[track_id] = 0

      laplacian_var = cv2.Laplacian(state['plate'], cv2.CV_64F).var()

      if laplacian_var >= self.blur_threshold:
        plate_text = self.reader.predict(state['plate'])
        self.voter.add_prediction(track_id, plate_text)
        
      final_plate_text = self.voter.get_final_plate(track_id)

      self._save_to_db(track_id, final_plate_text, state['conf'])

      state['text'] = final_plate_text
      frame_results[track_id] = state

    tracks_to_remove = []
    for track_id in self.missing_frames:
      if track_id not in frame_results:
        self.missing_frames[track_id] += 1

        if self.missing_frames[track_id] > 30:
          self.voter.clear_history(track_id)
          tracks_to_remove.append(track_id)
          self.saved_track.discard(track_id)

    for track_id in tracks_to_remove:
      del self.missing_frames[track_id]

    return frame_results
  
  def _save_to_db(self, track_id, plate_text, confidence):
    if track_id not in self.saved_track:
      if plate_text and plate_text != "UNKNOWN!" and "؟" not in plate_text:
        with get_db_context() as db:
          plate = DetectedPlate(track_id=track_id, plate_text=plate_text, confidence=confidence)
          db.add(plate)

        self.saved_track.add(track_id)

