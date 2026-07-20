from src.tracking.tracker import PlateTracker
from src.ocr.predictor import PlateReader
from src.decision.voting import TemporalVoter


class ANPRPipeline:
  def __init__(self, tracker_model_path, ocr_model_path, device=None):
    self.tracker = PlateTracker(tracker_model_path)
    self.reader = PlateReader(ocr_model_path, device)
    self.voter = TemporalVoter()

    self.missing_frames = {}

  def process_frame(self, frame):
    frame_results = {}
    
    current_frame_outputs = self.tracker.track(frame)
    for track_id, state in current_frame_outputs.items():
      self.missing_frames[track_id] = 0

      plate_text = self.reader.predict(state['plate'])

      self.voter.add_prediction(track_id, plate_text)
      final_plate_text = self.voter.get_final_plate(track_id)

      state['text'] = final_plate_text
      frame_results[track_id] = state

    tracks_to_remove = []
    for track_id in self.missing_frames:
      if track_id not in frame_results:
        self.missing_frames[track_id] += 1

        if self.missing_frames[track_id] > 30:
          self.voter.clear_history(track_id)
          tracks_to_remove.append(track_id)

    for track_id in tracks_to_remove:
      del self.missing_frames[track_id]

    return frame_results

