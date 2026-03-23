import streamlit as st
import pandas as pd
import csv
from PIL import Image
from ultralytics import YOLO
import cv2
import os


def estimate_yield():
    # 1. Setup paths and model
    output_csv = './output/yield_results2.csv'
    model_path = '../runs/detect/yield_estimation_project/orange_detection_v12/weights/best.pt'

    @st.cache_resource
    def load_model():
        return YOLO(model_path)

    model = load_model()

    st.title("Orange Yield Estimation Dashboard")
    confidence = st.slider("Set model confidence", 10, 100, 10)
    conf_percent = confidence / 100
    st.write(f"Model is {confidence}% confident")

    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files='directory' or True
    )

    if uploaded_files:
        # 2. Process images ONLY once (storing results in Session State)
        if 'processed_data' not in st.session_state or st.session_state.get('file_count') != len(uploaded_files):
            results_data = []

            with st.spinner('Running model inference...'):

                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)

                    # Run prediction
                    results = model.predict(image, conf=conf_percent, verbose=False)

                    #Run track
                    #results = model.track(image, conf=conf_percent, persist=True, verbose=False)

                    # Create the annotated image (boxes + scores)
                    annotated_img = results[0].plot()  # BGR array

                    # convert the BGR to RBG for clear visusals
                    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

                    boxes = results[0].boxes
                    count = len(boxes)

                    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
                    results_data.append({
                        'track_ids': track_ids,
                        'image_name': uploaded_file.name,
                        'estimated_yield': count,
                        'visual': annotated_rgb
                    })

            # Save to session state to prevent re-runs
            st.session_state.processed_data = results_data
            st.session_state.file_count = len(uploaded_files)
            st.session_state.current_idx = 0



        # 3. Define Tabs
        tab1, tab2 = st.tabs(["Orange Image", "Yield Report"])

        # --- TAB 1: Image Slider ---
        with tab1:
            data = st.session_state.processed_data

            # Navigation Controls
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous") and st.session_state.current_idx > 0:
                    st.session_state.current_idx -= 1
            with col2:
                st.write(f"Showing Image {st.session_state.current_idx + 1} of {len(data)}")
            with col3:
                if st.button("Next ➡️") and st.session_state.current_idx < len(data) - 1:
                    st.session_state.current_idx += 1

            # Current Selection Display
            current_item = data[st.session_state.current_idx]
            st.metric("Detected Count", f"{current_item['estimated_yield']} Oranges")
            st.write(f"File name: {current_item['image_name']}")

            # Show the image with boxes
            st.image(current_item['visual'], use_container_width=True)

        # --- TAB 2: Results Table ---
        with tab2:
            st.subheader("Yield Summary")

            # Create the DataFrame
            report_df = pd.DataFrame([
                {'Image Name': item['image_name'], 'Estimated Yield': item['estimated_yield'], 'Track IDs': item['track_ids']}
                for item in data
            ])

            # Calculate Summary Statistics
            total_oranges = report_df['Estimated Yield'].sum()
            avg_oranges = report_df['Estimated Yield'].mean()
            total_trees = len(report_df)

            # Display Metric Cards
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Yield", f"{total_oranges} Oranges", delta="Season Total")
            m2.metric("Avg. Yield / Tree", f"{avg_oranges:.1f}")
            m3.metric("Trees Surveyed", total_trees)

            st.markdown("---")
            left, right = st.columns(2, vertical_alignment='bottom')
            left.write("#### Detailed Inventory")
            #right.metric("Total Yield", f"{total_oranges} Oranges")

            # Professional Interactive Table
            st.dataframe(
                report_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Estimated Yield": st.column_config.NumberColumn(
                        "Estimated Yield",
                        help="Estimated number of oranges detected by the model",
                        format="%d "
                    )
                }
            )

            # Download Button (Cloud-Ready)
            csv_data = report_df.to_csv(index=False).encode('utf-8')
            right.download_button(
                label="Download Yield Report (CSV)",
                data=csv_data,
                file_name='orchard_yield_report.csv',
                mime='text/csv',

            )


if __name__ == "__main__":
    estimate_yield()