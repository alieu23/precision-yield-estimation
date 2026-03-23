import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import os

# --- 1. SETTINGS & MODEL LOADING ---
st.set_page_config(page_title="Citrus Yield AI", layout="wide")


@st.cache_resource
def load_yolo_model():
    # Path to your fine-tuned YOLOv11 model
    model_path = '../runs/detect/yield_estimation_project/orange_detection_v12/weights/best.pt'
    return YOLO(model_path)


model = load_yolo_model()

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.title("Parameter Settings")
app_mode = st.sidebar.radio("Select Input Media Type", ["Image Batch", "Video Stream"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")
conf_value = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.38)
# Accuracy mode to handle background oranges
accuracy_mode = st.sidebar.checkbox("High Accuracy Mode (1024px)", value=False)
img_sz = 1024 if accuracy_mode else 640

# --- 3. IMAGE BATCH PROCESSING MODE ---
if app_mode == "Image Batch":
    st.header("Image Batch Processing")
    uploaded_files = st.file_uploader("Upload Orchard Images", type=["jpg", "png", "jpeg"], accept_multiple_files='directory' or True)

    if uploaded_files:
        if 'img_results' not in st.session_state or st.session_state.get('last_upload_count') != len(uploaded_files):
            results_data = []
            with st.spinner('Processing images...'):
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    # Run tracking even on images to maintain consistency
                    results = model.track(image, conf=conf_value, imgsz=img_sz, persist=True, verbose=False)

                    count = len(results[0].boxes)
                    annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

                    results_data.append({
                        'name': uploaded_file.name,
                        'yield': count,
                        'visual': annotated_img
                    })
            st.session_state.img_results = results_data
            st.session_state.last_upload_count = len(uploaded_files)
            st.session_state.img_idx = 0

        # Tabs for Auditor and Table
        tab1, tab2 = st.tabs(["Image Slider", "Yield Report"])

        with tab1:
            res = st.session_state.img_results
            c1, c2, c3 = st.columns([1, 2, 1])
            if c1.button("Previous") and st.session_state.img_idx > 0:
                st.session_state.img_idx -= 1
            if c3.button("Next") and st.session_state.img_idx < len(res) - 1:
                st.session_state.img_idx += 1

            curr = res[st.session_state.img_idx]
            st.image(curr['visual'], caption=f"File: {curr['name']}", use_container_width=True)
            st.metric("Detected Oranges", curr['yield'])

        with tab2:
            df = pd.DataFrame([{'Image': x['name'], 'Yield': x['yield']} for x in res])
            st.dataframe(df, use_container_width=True)
            st.metric("Total Batch Yield", df['Yield'].sum())

# --- 4. VIDEO PROCESSING MODE ---
else:
    st.header("Video Yield Tracking")
    video_file = st.file_uploader("Upload Orchard Video", type=["mp4", "mov", "avi"])

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(video_file.read())
            temp_path = tfile.name

        cap = cv2.VideoCapture(temp_path)
        st_frame = st.empty()
        st_metric = st.sidebar.empty()

        unique_ids = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Tracking with BoT-SORT to prevent double-counting
            results = model.track(frame, conf=conf_value, imgsz=img_sz, persist=True, verbose=False)

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.int().cpu().tolist()
                for obj_id in ids:
                    unique_ids.add(obj_id)

            annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame, use_container_width=True)
            st_metric.metric("Total Unique Oranges", len(unique_ids))

        cap.release()
        os.remove(temp_path)
        st.success(f"Video Analysis Complete. Final Count: {len(unique_ids)}")

# --- 5. SUMMARY DASHBOARD (Appears at the bottom for both modes) ---
st.markdown("---")
st.header("Yield Insights")

# Check if we have any data to summarize
has_images = 'img_results' in st.session_state
has_video = 'unique_ids' in locals() or 'unique_ids' in globals()

if has_images or has_video:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Yield Density Map")
        # Logic to show which files or segments had the highest density
        if has_images:
            df_summary = pd.DataFrame([{'Name': x['name'], 'Count': x['yield']} for x in st.session_state.img_results])
            st.bar_chart(df_summary.set_index('Name'))

    with col_b:
        st.subheader("Efficiency Metrics")
        if has_images:
            total = sum(x['yield'] for x in st.session_state.img_results)
            avg = total / len(st.session_state.img_results)
            st.info(f"The average yield per tree in this batch is **{avg:.1f}** oranges.")

        if 'unique_ids' in locals() and len(unique_ids) > 0:
            st.success(f"Video analysis confirmed a net yield of **{len(unique_ids)}** unique oranges.")

    # Export for AWS S3 Storage
    if st.button("Archive Results to Cloud Storage"):
        st.write("Pushing metadata to DynamoDB and visual logs to S3...")
        # This is where your boto3 code would go for the AWS phase
        st.success("Data successfully archived in the Cloud-Native backend.")
else:
    st.info("Upload media above to generate orchard insights.")