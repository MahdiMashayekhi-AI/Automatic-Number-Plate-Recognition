import logging
from ultralytics import YOLO
from datetime import datetime
from src.config import KEYPOINT_MODEL_PATH
from src.logger import setup_logger


def fine_tune_pose():
  setup_logger()
  logger = logging.getLogger(__name__)

  logger.info("Loading pretrained pose model...")
  model = YOLO(KEYPOINT_MODEL_PATH)

  run_name = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
  logger.info(f"Starting fine-tuning: {run_name}")

  results = model.train(
    data="data/pose/data.yaml",
    epochs=50,
    batch=4,
    imgsz=960,
    lr0=0.001,
    optimizer="AdamW",
    patience=15,
    project="outputs/pose_runs",
    name=run_name,
    exist_ok=False,
    workers=0,
    # degrees=10.0,
    # shear=5.0,
    # perspective=0.0005
  )

  logger.info(f"Fine-tuning complete. Best weights saved in outputs/pose_runs/{run_name}/weights/best.pt")
  logger.info("Copy this file to outputs/checkpoints/pose/best.pt after verifying performance.")

if __name__ == "__main__":
    fine_tune_pose()