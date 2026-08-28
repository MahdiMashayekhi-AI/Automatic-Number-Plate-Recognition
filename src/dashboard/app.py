import cv2
import torch
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from src.pipeline import ANPRPipeline
from src.database.connection import get_db_context
from src.database.models import DetectedPlate


st.set_page_config(
  page_title="ANPR System 🚗",
  page_icon="🚗",
  layout="wide"
)

@st.cache_resource
def load_pipeline():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return ANPRPipeline("license_plate_keypoint.pt", "outputs/checkpoints/best.pt", device)

pipeline = load_pipeline()

st.title("Automatic Number Plate Recognition (ANPR) Dashboard 🚗")

tab1, tab2 = st.tabs(['Process Image | Video', 'View Detected Plates'])

with tab1:
  st.header("Upload Image or Video")
  uploaded_file = st.file_uploader("Upload an image or video file", type=['png', 'jpg', 'jpeg', 'mp4', 'avi'])

  if uploaded_file:
    file_extention = uploaded_file.name.split(".")[-1].lower()

    if file_extention in ['png', 'jpg', 'jpeg']:
      imarr = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
      image = cv2.imdecode(imarr, cv2.IMREAD_COLOR)

      results = pipeline.process_frame(image)

      for track_id, data in results.items():
        xmin, ymin , xmax, ymax = data['bbox']
        plate_text = data['text']
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, plate_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      st.image(image, "Processed image")
    elif file_extention in ['mp4', 'avi']:
      tfile = tempfile.NamedTemporaryFile(delete=False)
      tfile.write(uploaded_file.read())

      cap = cv2.VideoCapture(tfile.name)
      st_frame = st.empty()

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        results = pipeline.process_frame(frame)

        for track_id, data in results.items():
          xmin, ymin, xmax, ymax = data['bbox']
          plate_text = data['text']
          cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
          cv2.putText(frame, f"Id: {track_id} | {plate_text}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame, channels="RGB")

      cap.release()


with tab2:
  st.header("Plate that recoreded in Database")

  if st.button("Refresh"):
    st.rerun()

  with get_db_context() as db:
    query = db.query(DetectedPlate).order_by(DetectedPlate.created_at.desc()).all()

    if query:
      data = [
        {
          "id": q.id,
          "track_id": q.track_id,
          "plate_text": q.plate_text,
          "confidence": f"{q.confidence:.2f}",
          "created_at": q.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for q in query
      ]

      df = pd.DataFrame(data)

      col1, col2 = st.columns(2)
      col1.metric("Number of plates recognized", len(df))

      st.dataframe(df)

      search_query = st.text_input("🔍 Searching for plate number:")
      if search_query:
        filtered_df = df[df["plate_text"].str.contains(search_query)]
        st.write(f"Results for {search_query}")
        st.dataframe(filtered_df)

    else:
      st.info("There is no data in database!")
