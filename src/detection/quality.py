import cv2
from src.config import BLUR_THRESHOLD


def compute_sharpness(image):
  if len(image.shape) == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  return cv2.Laplacian(image, cv2.CV_64F).var()

