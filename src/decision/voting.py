from collections import Counter
from src.config import MIN_VOTES_TO_CONFIRM
from src.decision.validator import is_valid_plate_format


class TemporalVoter:
  def __init__(self):
    self.history = {}

  def add_prediction(self, track_id, plate_text):
    if isinstance(plate_text, list):
      plate_text = "".join(plate_text)

    if not plate_text:
      return

    if not is_valid_plate_format(plate_text):
      return

    if track_id not in self.history:
        self.history[track_id] = []
    
    self.history[track_id].append(plate_text)

  def get_final_plate(self, track_id):
    if track_id not in self.history or not self.history[track_id]:
        return None
    
    cnt = Counter(self.history[track_id])
    most_common_items = cnt.most_common(1)

    if not most_common_items:
      return None

    most_common, count = most_common_items[0]

    if count >= MIN_VOTES_TO_CONFIRM:
      return most_common

    return None

  def get_best_guess(self, track_id):
    if track_id not in self.history or not self.history[track_id]:
      return None

    cnt = Counter(self.history[track_id])
    most_common_items = cnt.most_common(1)

    if not most_common_items:
      return None

    most_common, _ = most_common_items[0]

    return most_common
  
  def clear_history(self, track_id):
    if track_id in self.history:
      del self.history[track_id]