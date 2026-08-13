import os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from src.logger import setup_logger
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.ocr.dataset import PlateDataset
from src.config import BATCH_SIZE, CHAR_LIST, LEARNING_RATE, EPOCHS, IDX2CHAR
from src.ocr.model import CRNN
from src.ocr.utils import ctc_decode, calculate_metrics


def train():
  setup_logger()
  logger = logging.getLogger(__name__)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info(f"Device is {device}")
  
  train_dataset = PlateDataset("data/raw/", "train_labels.txt", None)
  val_dataset = PlateDataset("data/raw/", "val_labels.txt", None)

  train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

  model = CRNN(input_channel=1, num_classes=len(CHAR_LIST), hidden_size=256)
  model.to(device)

  criterion = nn.CTCLoss(blank=0, zero_infinity=True)
  optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

  best_sequence_accuracy = 0

  for epoch in range(EPOCHS):
    total_loss = 0
    total_samples = 0
    
    model.train()
    for image, label, label_len in tqdm(
      train_loader,
      desc=f"Epoch {epoch + 1}/{EPOCHS}"):
      image, label = image.to(device), label.to(device)

      optimizer.zero_grad()

      # Forward pass
      outputs = model(image)

      log_probs = F.log_softmax(outputs, dim=2)

      input_length = torch.full((image.size(0),), log_probs.size(0), dtype=torch.long, device=device)

      loss = criterion(log_probs, label, input_length, label_len)
      loss.backward()

      total_loss += loss.item() * image.size(0)
      total_samples += image.size(0) 

      nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=5.0)

      optimizer.step()

    avg_loss = total_loss / total_samples
    logger.info(f"Epoch {epoch + 1}, Train Loss: {avg_loss}")

    total_loss = 0
    total_samples = 0

    total_correct = 0
    total_sequences = 0
    total_distance = 0
    total_chars = 0

    model.eval()
    with torch.no_grad():
      for image, label, label_len in tqdm(
          val_loader,
          desc=f"Validation {epoch + 1}/{EPOCHS}"
      ):
        image, label = image.to(device), label.to(device)

        outputs = model(image)

        log_probs = F.log_softmax(outputs, dim=2)

        input_length = torch.full((image.size(0),), log_probs.size(0), dtype=torch.long, device=device)

        loss = criterion(log_probs, label, input_length, label_len)

        total_loss += loss.item() * image.size(0)
        total_samples += image.size(0)

        preds = ctc_decode(log_probs)
        targets = []
        for lbl, length in zip(label, label_len):
          targets.append(''.join([IDX2CHAR[c.item()] for c in lbl[:length.item()]]))

        correct, samples, distance, chars = calculate_metrics(preds, targets)

        total_correct += correct
        total_sequences += samples
        total_distance += distance
        total_chars += chars

    avg_loss = total_loss / total_samples
    logger.info(f"Epoch {epoch + 1}, Validation Loss: {avg_loss}")

    sequence_accuracy = total_correct / total_sequences
    cer = total_distance / total_chars

    logger.info(f"Sequence Accuracy: {sequence_accuracy*100:.2f}%, CER: {cer*100:.2f}%")

    if not os.path.exists("outputs"):
      os.mkdir("outputs")

    if sequence_accuracy > best_sequence_accuracy:
      torch.save(model.state_dict(), "outputs/best_model.pt")
      best_sequence_accuracy = sequence_accuracy

    sample_pred = ctc_decode(log_probs.detach())[0]
    sample_target = ''.join([IDX2CHAR[c.item()] for c in label[0][:label_len[0].item()]])
    logger.info(f"Pred: {sample_pred} | GT: {sample_target}")


if __name__ == "__main__":
  train()