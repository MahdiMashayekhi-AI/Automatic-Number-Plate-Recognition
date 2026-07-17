from ultralytics import YOLO
from src.detection.geometry import crop_and_deskew


class PlateDetector:
  def __init__(self, model_path):
    self.model = YOLO(model_path)

  def detect(self, image):
    results = self.model(image, conf=0.4)

    deskewed_plates = []

    for result in results:
      if result.boxes is None:
        continue

      boxes = result.boxes
      keypoints = result.keypoints

      for box, kp in zip(boxes, keypoints):
        
        score = box.conf[0].item()

        points = kp.xy.cpu().numpy()[0]

        plate = crop_and_deskew(image, points)
        deskewed_plates.append({
          "score": score,
          "keypoints": points,
          "image": plate,
        })

    return deskewed_plates

