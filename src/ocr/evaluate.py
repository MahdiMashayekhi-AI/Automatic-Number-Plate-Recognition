import os
import torch
import logging
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from src.logger import setup_logger
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.ocr.dataset import PlateDataset
from src.config import BATCH_SIZE, CHAR_LIST, IDX2CHAR
from src.ocr.model import CRNN
from src.ocr.utils import ctc_decode, calculate_metrics


def evaluate():
  setup_logger()
  logger = logging.getLogger(__name__)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info(f"Device is {device}")

  test_dataset = PlateDataset("data/raw", 'test_labels.txt', return_path=True)

  test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

  model = CRNN(input_channel=1, num_classes=len(CHAR_LIST), hidden_size=256)
  checkpoint = torch.load("outputs/checkpoints/best.pt", map_location=device)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.to(device)
  model.eval()

  total_loss = 0
  total_samples = 0
  total_correct = 0
  total_sequences = 0
  total_distance = 0
  total_chars = 0
  failure_cases = []

  criterion = nn.CTCLoss(blank=0, zero_infinity=True)

  with torch.no_grad():
    for image, label, label_len, image_path in tqdm(test_loader, desc=f'Test Dataset'):
      image, label = image.to(device), label.to(device)

      outputs = model(image)

      log_probs = F.log_softmax(outputs, dim=2)

      input_length = torch.full((image.size(0), ), log_probs.size(0), dtype=torch.long, device=device)

      loss = criterion(log_probs, label, input_length, label_len)

      total_loss += loss.item() * image.size(0)
      total_samples += image.size(0)

      preds = ctc_decode(log_probs)

      targets = []
      for lbl, length in zip(label, label_len):
        targets.append(''.join([IDX2CHAR[c.item()] for c in lbl[:length.item()]]))

      for pred, target, path in zip(preds, targets, image_path):
        if pred != target:
          failure_cases.append({"pred": pred, "target": target, "image_path": path})
      
      correct, samples, distance, chars = calculate_metrics(preds, targets)

      total_correct += correct
      total_sequences += samples
      total_distance += distance
      total_chars += chars

    sequence_accuracy = total_correct / total_sequences
    cer = total_distance / total_chars
    avg_loss = total_loss / total_samples

    logger.info(f"Test Sequence Accuracy: {sequence_accuracy * 100:.2f}%")
    logger.info(f"Test CER: {cer * 100:.2f}%")
    logger.info(f"Test Loss: {avg_loss:.4f}")
    logger.info(f"Test samples: {len(test_dataset)}")

    save_failure_cases_path = "outputs/evaluation/"
    if not os.path.exists(save_failure_cases_path):
      os.mkdir(save_failure_cases_path)

    df = pd.DataFrame(failure_cases)
    df.to_csv(os.path.join(save_failure_cases_path, "test_failures.csv"), index=False, encoding="utf-8-sig")
    logger.info("Successfuly saved failure cases.")

if __name__ == "__main__":
  evaluate()