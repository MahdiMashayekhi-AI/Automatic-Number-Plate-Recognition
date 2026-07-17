import torch
import torch.nn as nn


class CRNN(nn.Module):
  def __init__(self, input_channel, num_classes, hidden_size=256):
    super().__init__()

    self.cnn = nn.Sequential(
      # Block 1
      nn.Conv2d(input_channel, 64, 3, 1, 1),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.Dropout2d(0.2),
      nn.MaxPool2d(kernel_size=2, stride=2),

      # Block 2
      nn.Conv2d(64, 128, 3, 1, 1),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.Dropout2d(0.2),
      nn.MaxPool2d(kernel_size=2, stride=2),

      # Block 3
      nn.Conv2d(128, 256, 3, 1, 1),
      nn.BatchNorm2d(256),
      nn.ReLU(True),
      nn.Dropout2d(0.2),
      nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

      # Block 4
      nn.Conv2d(256, 256, 3, 1, 1),
      nn.BatchNorm2d(256),
      nn.ReLU(True),
      nn.Dropout2d(0.2),
      nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

      # Block 5
      nn.Conv2d(256, 512, kernel_size=(2, 1), padding=0),
      nn.BatchNorm2d(512),
      nn.Dropout2d(0.2),
      nn.ReLU(True)
    )

    self.rnn = nn.LSTM(512, hidden_size, num_layers=2, batch_first=False, bidirectional=True, dropout=0.2)

    self.fc = nn.Sequential(
      nn.Linear(hidden_size*2, 256),
      nn.ReLU(True),
      nn.Dropout(0.4),
      nn.Linear(256, num_classes)
    )

    self.apply(self._init_weights)

  def forward(self, x):
    # Input Shape = [Batch, 1, 32, 128] -> Channel, Height, Width
    x = self.cnn(x)

    # Change the Shape -> [Sequence_length, Batch_size, Features]
    x = x.squeeze(2)
    x = x.permute(2, 0, 1).contiguous()

    x, _ = self.rnn(x)

    return self.fc(x)
  
  def _init_weights(self, m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight, 1),
      nn.init.constant_(m.bias, 0)
