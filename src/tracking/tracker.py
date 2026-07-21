from ultralytics import YOLO
from src.detection.geometry import crop_and_deskew


class PlateTracker:
  def __init__(self, model_path):
    self.model = YOLO(model_path)
    self.track_states = {}


  def track(self, frame):
    current_active_tracks = set()
    current_frame_output = {}

    results = self.model.track(frame, tracker='bytetrack.yaml', persist=True, conf=0.4)

    for result in results:
      boxes = result.boxes
      track_ids = boxes.id
      keypoints = result.keypoints

      if track_ids is not None:
        for box, track_id, kp in zip(boxes, track_ids, keypoints):
          xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
          kp = kp.xy.cpu().numpy()[0]

          deskewed_plate = crop_and_deskew(frame, kp)
          conf = float(box.conf[0]) 
          track_id = int(track_id)

          current_active_tracks.add(track_id)

          if track_id not in self.track_states:
            self.track_states[track_id] = {
              "conf": conf,
              "plate": deskewed_plate,
              # "bbox": [xmin, ymin, xmax, ymax],
              "missing_frames": 0
            }

          else:
            if self.track_states[track_id]['conf'] < conf:
              self.track_states[track_id]["conf"] = conf
              self.track_states[track_id]["plate"] = deskewed_plate
              # self.track_states[track_id]["bbox"] = [xmin, ymin, xmax, ymax]
              # self.track_states[track_id]["missing_frames"] = 0

          current_frame_output[track_id] = {
            "conf": conf,
            "plate": self.track_states[track_id]["plate"],
            "bbox": [xmin, ymin, xmax, ymax]
          }


    tracks_to_remove = []
    for track_id in self.track_states:
      if track_id in current_active_tracks:
        self.track_states[track_id]["missing_frames"] = 0
      else:
        self.track_states[track_id]["missing_frames"] += 1
      
      if self.track_states[track_id]["missing_frames"] > 30:
        tracks_to_remove.append(track_id)      

    for track_id in tracks_to_remove:
      del self.track_states[track_id]
    
    return current_frame_output