import os
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from src.config import CHAR2IDX, IDX2CHAR, IMAGE_WIDTH, IMAGE_HEIGHT


class PlateDataset(Dataset):
  def __init__(self, root_dir, label_file, transform=None):
    self.root_dir = root_dir
    self.df = pd.read_csv(os.path.join(self.root_dir, label_file), sep=" ", header=None, names=['image_path', 'label'])

    if transform is None:
      self.transform = transforms.ToTensor()
    else:
      self.transform = transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    row = self.df.iloc[index]
    image_path = row['image_path']
    label = row['label']

    img_array = np.fromfile(os.path.join(self.root_dir, image_path), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    if image is None:
      raise FileNotFoundError(f"Path of the image is incorrect! {os.path.join(self.root_dir, image_path)}")

    image_resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    normalized = self.transform(image_resized)

    numerical_label = torch.LongTensor([CHAR2IDX[letter] for letter in label])
    label_len = len(label)

    return normalized, numerical_label, label_len

