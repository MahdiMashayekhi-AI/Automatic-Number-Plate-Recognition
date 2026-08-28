from ultralytics import YOLO
from src.detection.geometry import crop_and_deskew
from src.detection.quality import compute_sharpness
from src.config import BLUR_THRESHOLD


class PlateTracker:
  def __init__(self, model_path):
    self.model = YOLO(model_path)
    self.track_states = {}


  def track(self, frame):
    current_active_tracks = set()
    current_frame_output = {}

    results = self.model.track(frame, tracker='custom_tracker.yaml', persist=True, conf=0.3, imgsz=960)

    for result in results:
      boxes = result.boxes
      track_ids = boxes.id
      keypoints = result.keypoints

      if track_ids is not None:
        for box, track_id, kp in zip(boxes, track_ids, keypoints):
          xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
          kp = kp.xy.cpu().numpy()[0]

          deskewed_plate = crop_and_deskew(frame, kp)
          if deskewed_plate is None:
            continue
          
          conf = float(box.conf[0]) 
          track_id = int(track_id)

          current_active_tracks.add(track_id)

          new_sharpness = compute_sharpness(deskewed_plate)
          new_conf = conf

          if track_id not in self.track_states:
            self.track_states[track_id] = {
              'conf': new_conf,
              'sharpness': new_sharpness,
              'plate': deskewed_plate,
              'missing_frames': 0
            }
          else:
            current = self.track_states[track_id]
            new_is_sharp = new_sharpness >= BLUR_THRESHOLD
            current_is_sharp = current['sharpness'] >= BLUR_THRESHOLD

            should_update = False

            if new_is_sharp and not current_is_sharp:
              should_update = True
            elif new_is_sharp and current_is_sharp:
              if new_conf > current['conf']:
                should_update = True
            elif not new_is_sharp and not current_is_sharp:
              if new_sharpness > current['sharpness']:
                should_update = True

            if should_update:
              current['conf'] = new_conf
              current['sharpness'] = new_sharpness
              current['plate'] = deskewed_plate

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