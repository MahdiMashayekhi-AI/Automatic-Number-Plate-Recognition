import cv2
import torch
from src.pipeline import ANPRPipeline


def main():
  TRACKER_MODEL = "license_plate_keypoint.pt"
  OCR_MODEL = "outputs/best_model.pt"
  VIDEO_PATH = "./data/samples/Tehran-Traffic.mp4"
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print("[INFO] Loading ANPR Pipeline...")
  pipeline = ANPRPipeline(TRACKER_MODEL, OCR_MODEL, DEVICE)

  cap = cv2.VideoCapture(VIDEO_PATH)
  if not cap.isOpened():
    print(f"[ERROR] Could not open video: {VIDEO_PATH}")
    return
  
  print("[INFO] Starting video processing. Press 'q' to quit.")
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    results = pipeline.process_frame(frame)

    for track_id, data in results.items():
      xmin, ymin, xmax, ymax = data['bbox']
      plate_text = data['text']
      conf = data['conf']

      cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

      label = f"ID: {track_id} | {plate_text} ({conf:.2f})"
      cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("ANPR Production Test", frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
  print("[INFO] Processing finished successfully.")


if __name__ == "__main__":
  main()