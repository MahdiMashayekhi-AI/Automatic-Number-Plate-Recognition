from collections import Counter


class TemporalVoter:
  def __init__(self):
    self.history = {}

  def add_prediction(self, track_id, plate_text):
    if isinstance(plate_text, list):
      plate_text = "".join(plate_text)

    if not plate_text:
      plate_text = "UNKNOWN!"

    if len(plate_text) < 8:
      plate_text = plate_text.ljust(8, "?")
    elif len(plate_text) > 8:
      plate_text = plate_text[:8]

    if track_id not in self.history:
      self.history[track_id] = []
    self.history[track_id].append(plate_text)

  def get_final_plate(self, track_id):
    if track_id in self.history and self.history[track_id]:
      if len(self.history[track_id]) != 0:
        final_plate = []
        for index in range(8):
          cols_list = [text[index] for text in self.history[track_id]]
          cnt = Counter(cols_list)
          char = max(cnt, key=cnt.get)
          final_plate.append(char)
        
        return "".join(final_plate)
    
    return "UNKOWN!"
  
  def clear_history(self, track_id):
    if track_id in self.history:
      del self.history[track_id]