import torch
import Levenshtein as l
from src.config import IDX2CHAR


def ctc_decode(outputs):
  prediction = torch.argmax(outputs, dim=2)
  prediction = prediction.permute(1, 0).contiguous()

  texts = []
  for seq in prediction:
    prev = None
    text = []
    for idx in seq:
      idx = idx.item()
      if idx != 0:
        if idx != prev:
          text.append(IDX2CHAR[idx])
      prev = idx

    texts.append(''.join(text))

  return texts


def calculate_accuracy(preds, targets):
  if isinstance(preds, str): preds = [preds]
  if isinstance(targets, str): targets = [targets]
  
  accuracy = 0
  total_distance = 0  
  total_chars = 0

  for pred, target in zip(preds, targets):
    accuracy += 1 if pred == target else 0

    distance = l.distance(pred, target)
    total_distance += distance
    total_chars += len(target)

  acc = accuracy / len(preds)
  cer = total_distance / total_chars

  return acc, cer