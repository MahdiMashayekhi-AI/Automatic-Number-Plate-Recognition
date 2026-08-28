import cv2
from src.tracking.tracker import PlateTracker
from src.detection.detector import PlateDetector
from src.detection.quality import compute_sharpness
from src.decision.validator import is_valid_plate_format
from src.ocr.predictor import PlateReader
from src.decision.voting import TemporalVoter
from src.database.connection import get_db_context, Base, engine
from src.database.models import DetectedPlate
from src.config import BLUR_THRESHOLD


class ANPRPipeline:
  def __init__(self, tracker_model_path, ocr_model_path, device=None, blur_threshold=BLUR_THRESHOLD):
    Base.metadata.create_all(bind=engine)

    self.tracker = PlateTracker(tracker_model_path)
    self.detector = PlateDetector(tracker_model_path)
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

      sharpness = compute_sharpness(state['plate'])

      if sharpness >= self.blur_threshold:
        plate_text = self.reader.predict(state['plate'])
        self.voter.add_prediction(track_id, plate_text)
        
      final_plate_text = self.voter.get_final_plate(track_id)
      best_guess = self.voter.get_best_guess(track_id)

      if final_plate_text is not None:
        if track_id not in self.saved_track:
          self._save_to_db(track_id, final_plate_text, state['conf'])
        state['text'] = final_plate_text
      elif best_guess is not None:
        state['text'] = best_guess
      else:
        state['text'] = "Detecting"

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

  def process_image(self, image):
    results = self.detector.detect(image)

    outputs = []
    for result in results:
      sharpness  = compute_sharpness(result['image'])
      is_sharp = sharpness >= self.blur_threshold

      if not is_sharp:
        outputs.append({
          "plate_text": None,
          "is_confident": False,
          "reason": "low_sharpness",
          "score": result['score'],
          "bbox": result['bbox']
        })
        continue

      raw_prediction = self.reader.predict(result['image'])
      if isinstance(raw_prediction, list):
        raw_prediction = "".join(raw_prediction)

      is_valid = is_valid_plate_format(raw_prediction)

      outputs.append({
        "plate_text": raw_prediction,
        "is_confident": is_valid,
        "reason": None if is_valid else "invalid_format",
        "score": result['score'],
        "bbox": result['bbox']
      })

    return outputs

  def reset(self):
    self.tracker.track_states = {}
    self.voter.history = {}
    self.missing_frames = {}
    self.saved_track = set()
  
  def _save_to_db(self, track_id, plate_text, confidence):
    if track_id not in self.saved_track:
        with get_db_context() as db:
          plate = DetectedPlate(track_id=track_id, plate_text=plate_text, confidence=confidence)
          db.add(plate)

        self.saved_track.add(track_id)

