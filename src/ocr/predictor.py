import cv2
import torch
import torch.nn.functional as F
from src.ocr.model import CRNN
from src.config import CHAR_LIST
from src.ocr.utils import ctc_decode


class PlateReader:
  def __init__(self, model_path, device=None):
    self.device = device
    if device is None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    self.model = CRNN(1, len(CHAR_LIST), 256) 
    self.model.load_state_dict(torch.load(model_path, map_location=self.device)) 
    self.model.to(self.device)  
    self.model.eval()

  def preprocessor(self, plate_img):
    if len(plate_img.shape) == 3:
      plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    plate_img = plate_img.astype(float) / 255

    plate_img = torch.FloatTensor(plate_img).to(self.device)

    return plate_img.unsqueeze(0).unsqueeze(0)
  
  def predict(self, plate_img):
    plate_img = self.preprocessor(plate_img)

    with torch.no_grad():
      outputs = self.model(plate_img)

      log_probs = F.log_softmax(outputs, dim=2)

      return ctc_decode(log_probs)