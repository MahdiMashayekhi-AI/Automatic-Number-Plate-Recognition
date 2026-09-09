import cv2
import torch
import logging
from src.logger import setup_logger
from src.pipeline import ANPRPipeline
from src.config import KEYPOINT_MODEL_PATH, OCR_MODEL_PATH, VIDEO_PATH, PROCESS_EVERY_N_FRAMES


def main():
  setup_logger()

  logger = logging.getLogger(__name__)

  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  logger.info("Loading ANPR Pipeline...")
  pipeline = ANPRPipeline(KEYPOINT_MODEL_PATH, OCR_MODEL_PATH, DEVICE)

  cap = cv2.VideoCapture(VIDEO_PATH)
  if not cap.isOpened():
    logger.error(f"Could not open video: {VIDEO_PATH}")
    return

  frame_counter = 0
  last_results = {}
  
  logger.info("Starting video processing. Press 'q' to quit.")
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    frame_counter += 1

    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
      last_results = pipeline.process_frame(frame)

    for track_id, data in last_results.items():
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
  logger.info("Processing finished successfully.")


if __name__ == "__main__":
  main()