import cv2
import numpy as np


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

    destination_points = np.array([[0, 0], [128, 0], [128, 32], [0, 32]], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(ordered_points, destination_points)

    return cv2.warpPerspective(image, matrix, (128, 32))