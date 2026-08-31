import cv2
import torch
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from src.pipeline import ANPRPipeline
from src.database.connection import get_db_context
from src.database.models import DetectedPlate
from src.config import KEYPOINT_MODEL_PATH, OCR_MODEL_PATH
from src.display.formatting import change_fa_to_en


st.set_page_config(
    page_title="ANPR System 🚗",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ANPRPipeline(KEYPOINT_MODEL_PATH, OCR_MODEL_PATH, device)

with st.spinner("🚀 Initializing AI Models & Pipeline..."):
    pipeline = load_pipeline()

with st.sidebar:
    st.title("⚙️ Control Panel")
    st.caption("AI-Powered Automatic Number Plate Recognition for Iranian Plates.")
    
    st.markdown("---")
    st.subheader("📂 Data Input")
    uploaded_file = st.file_uploader(
        "Upload Image/Video Source", 
        type=['png', 'jpg', 'jpeg', 'mp4', 'avi'],
        help="Select a single frame image or a video stream"
    )
    
    st.markdown("---")
    st.subheader("🖥️ Hardware Info")
    device_name = "NVIDIA CUDA GPU 🚀" if torch.cuda.is_available() else "CPU Mode 🐢"
    st.info(f"Running Inference on:\n**{device_name}**")

st.title("⚡ ANPR Real-Time Analytics Dashboard")
st.caption("Advanced Computer Vision System for High-Speed Plate Detection & OCR")
st.markdown("---")

tab1, tab2 = st.tabs(['📊 Live Inference Stream', '🗄️ Database Records'])

with tab1:
    if not uploaded_file:
        st.info("👈 Please upload an image or video from the sidebar to start recognition.")
    else:
        file_extention = uploaded_file.name.split(".")[-1].lower()

        if file_extention in ['png', 'jpg', 'jpeg']:
            imarr = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(imarr, cv2.IMREAD_COLOR)

            with st.spinner("Processing Image..."):
                results = pipeline.process_image(image)

            for result in results:
                xmin, ymin, xmax, ymax = result['bbox']
                plate_text = result['plate_text']
                color = (0, 255, 0) if result['is_confident'] else (0, 165, 225)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
                cv2.putText(image, plate_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            total_det = len(results)
            valid_det = sum(1 for r in results if r.get('is_confident', False))
            avg_conf = (sum(r['score'] for r in results) / total_det * 100) if total_det > 0 else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Detected Plates", total_det)
            m2.metric("Valid Format Plates", valid_det)
            m3.metric("Avg Confidence", f"{avg_conf:.1f}%")

            st.markdown("---")
            col_main, col_side = st.columns([2.5, 1])

            with col_main:
                st.subheader("🖼️ Processed Output")
                st.image(image, use_container_width=True)

            with col_side:
                st.subheader("🎯 Detected Plates")
                if not results:
                    st.warning("No License Plates Detected")
                else:
                    with st.container(height=500, border=False):
                        for i, result in enumerate(results):
                            with st.container(border=True):
                                plate_image = cv2.cvtColor(result['plate_image'], cv2.COLOR_BGR2RGB)
                                st.image(plate_image, use_container_width=True)
                                
                                plate_str = result['plate_text'] if result['plate_text'] else 'Unknown'
                                st.markdown(f"### `{change_fa_to_en(plate_str)}`")
                                
                                if result['is_confident']:
                                    st.success("✅ Valid Plate Format")
                                else:
                                    st.warning(f"⚠️ {result['reason']}")

                                st.caption(f"Confidence: **{result['score']*100:.1f}%**")

        elif file_extention in ['mp4', 'avi']:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            pipeline.reset()
            plate_history = {}
            last_snapshot = None

            try:
                cap = cv2.VideoCapture(tfile.name)

                kpi1, kpi2, kpi3 = st.columns(3)
                metric_total = kpi1.empty()
                metric_valid = kpi2.empty()
                metric_conf = kpi3.empty()

                st.markdown("---")
                col_video, col_history = st.columns([2.5, 1])

                with col_video:
                    st.subheader("📹 Video Stream")
                    st_frame = st.empty()

                with col_history:
                    st.subheader("📜 Live Recognition Log")
                    history_placeholder = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = pipeline.process_frame(frame)

                    for track_id, data in results.items():
                        xmin, ymin, xmax, ymax = data['bbox']
                        plate_text = data['text']
                        color = (0, 255, 0) if data['is_confident'] else (0, 165, 225)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
                        cv2.putText(frame, f"ID: {track_id} | {plate_text}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        plate_history[track_id] = {
                            "plate_image": data['plate'],
                            "plate_text": data['text'],
                            "conf": data['conf'],
                            "is_confident": data['is_confident'],
                            "reason": data['reason']
                        }

                    t_count = len(plate_history)
                    v_count = sum(1 for item in plate_history.values() if item['is_confident'])
                    c_avg = (sum(item['conf'] for item in plate_history.values()) / t_count * 100) if t_count > 0 else 0.0

                    metric_total.metric("Unique Plates Detected", t_count)
                    metric_valid.metric("Valid Format Count", v_count)
                    metric_conf.metric("Mean Confidence", f"{c_avg:.1f}%")

                    current_snapshot = tuple((k, v['plate_text']) for k, v in plate_history.items())

                    if current_snapshot != last_snapshot:
                        last_snapshot = current_snapshot

                        with history_placeholder.container(height=550, border=True):
                            if not plate_history:
                                st.info("Scanning for vehicle plates...")
                            else:
                                for track_id, item in reversed(list(plate_history.items())):
                                    with st.container(border=True):
                                        plate_image = cv2.cvtColor(item['plate_image'], cv2.COLOR_BGR2RGB)
                                        st.image(plate_image, use_container_width=True)
                                        
                                        plate_txt = item['plate_text'] if item['plate_text'] else 'Scanning...'
                                        st.markdown(f"**ID [{track_id}]:** `{change_fa_to_en(plate_txt)}`")

                                        if item['is_confident']:
                                            st.success("Valid", icon="✅")
                                        else:
                                            st.warning(f"{item['reason']}", icon="⚠️")

                                        st.caption(f"Score: **{item['conf']*100:.1f}%**")

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame, channels="RGB", use_container_width=True)

                cap.release()

            finally:
                pipeline.reset()
                if 'cap' in locals() and cap.isOpened():
                    cap.release()

with tab2:
    st.subheader("🗄️ Historical Plate Database")

    c1, c2 = st.columns([4, 1])
    with c2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    with get_db_context() as db:
        query = db.query(DetectedPlate).order_by(DetectedPlate.created_at.desc()).all()

        if query:
            data = [
                {
                    "ID": q.id, 
                    "Track ID": q.track_id,
                    "Plate Text": q.plate_text,
                    "Confidence": f"{q.confidence * 100:.1f}%",
                    "Timestamp": q.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "Image Path": q.image_path
                }
                for q in query
            ]

            df = pd.DataFrame(data)

            with c1:
                search_query = st.text_input("🔍 Search License Plate", placeholder="Type plate number e.g. 12B345...")

            if search_query:
                df = df[df["Plate Text"].str.contains(search_query, na=False)]

            st.markdown("---")
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Plate Text": st.column_config.TextColumn("Plate Text", help="OCR Extracted Text"),
                }
            )

        else:
            st.info("No detections recorded in the database yet.")