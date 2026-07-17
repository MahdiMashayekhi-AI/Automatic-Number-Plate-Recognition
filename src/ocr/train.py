import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from src.ocr.dataset import PlateDataset
from src.config import BATCH_SIZE, CHAR_LIST, LEARNING_RATE, EPOCHS, IDX2CHAR
from src.ocr.model import CRNN
from src.ocr.utils import ctc_decode, calculate_accuracy


def train():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  train_dataset = PlateDataset("data/raw/", "train_labels.txt", None)
  test_dataset = PlateDataset("data/raw/", "test_labels.txt", None)

  train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
  test_loader = DataLoader(test_dataset, BATCH_SIZE, False)

  model = CRNN(1, len(CHAR_LIST), 256)
  model.to(device)

  criterion = nn.CTCLoss(blank=0, zero_infinity=True)
  optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

  min_acc = 0

  for epoch in range(EPOCHS):
    total_loss = 0
    total_samples = 0
    
    model.train()
    for image, label, label_len in train_loader:
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
    print(f"Epoch {epoch + 1}, Train Loss: {avg_loss}")

    total_loss = 0
    total_samples = 0

    acc_list = []
    cer_list = []

    model.eval()
    with torch.no_grad():
      for image, label, label_len in test_loader:
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

        acc, cer = calculate_accuracy(preds, targets)
        acc_list.append(acc)
        cer_list.append(cer)

    avg_loss = total_loss / total_samples
    print(f"Epoch {epoch + 1}, Test Loss: {avg_loss}")

    avg_acc = sum(acc_list) / len(acc_list)
    avg_cer = sum(cer_list) / len(cer_list)

    print(f"Accuracy: {avg_acc}, CER: {avg_cer}")

    if not os.path.exists("outputs"):
      os.mkdir("outputs")

    if avg_acc > min_acc:
      torch.save(model.state_dict(), "outputs/best_model.pt")
      min_acc = avg_acc

    sample_pred = ctc_decode(log_probs.detach())[0]
    sample_target = ''.join([IDX2CHAR[c.item()] for c in label[0][:label_len[0].item()]])
    print(f"Pred: {sample_pred} | GT: {sample_target}")


if __name__ == "__main__":
  train()