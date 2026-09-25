import cv2
import math
import numpy as np
from src.config import IMAGE_HEIGHT, IMAGE_WIDTH, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO, MIN_QUAD_AREA


def order_points(points):
  ordered_points = np.zeros((4, 2), dtype=np.float32)

  sum_coords = points.sum(axis=1)
  diff_coords = points[:, 1] - points[:, 0]

  ordered_points[0] = points[np.argmin(sum_coords)]
  ordered_points[1] = points[np.argmin(diff_coords)]
  ordered_points[2] = points[np.argmax(sum_coords)]
  ordered_points[3] = points[np.argmax(diff_coords)]

  return ordered_points


def crop_and_deskew(image, keypoints):
    ordered_points = order_points(keypoints)

    if not is_valid_geometry(ordered_points):
      return None

    destination_points = np.array([[0, 0], [IMAGE_WIDTH, 0], [IMAGE_WIDTH, IMAGE_HEIGHT], [0, IMAGE_HEIGHT]], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(ordered_points, destination_points)

    return cv2.warpPerspective(image, matrix, (IMAGE_WIDTH, IMAGE_HEIGHT))


def compute_quad_area(ordered_points):
   x = ordered_points[:, 0]
   y = ordered_points[:, 1]

   s1 = np.dot(x, np.roll(y, -1))
   s2 = np.dot(y, np.roll(x, -1))

   return 0.5 * np.abs(s1 - s2)


def compute_aspect_ratio(ordered_points):
  upper_width = math.dist(ordered_points[0], ordered_points[1])
  lower_width = math.dist(ordered_points[3], ordered_points[2])
  average_width = (upper_width + lower_width) / 2

  left_height = math.dist(ordered_points[0], ordered_points[3])
  right_height = math.dist(ordered_points[1], ordered_points[2])
  average_height = (left_height + right_height) / 2

  if average_height < 1e-6:
    return 9999.0

  return average_width / average_height


def is_valid_geometry(ordered_points):
  area = compute_quad_area(ordered_points)
  if area < MIN_QUAD_AREA:
    # return False
    print("Skipping in geometry!")

  ratio = compute_aspect_ratio(ordered_points)
  if ratio < MIN_ASPECT_RATIO or ratio > MAX_ASPECT_RATIO:
    # return False
    print("Skipping in geometry!")

  return True