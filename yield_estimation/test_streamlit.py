import streamlit as st
import csv
from PIL import Image
from ultralytics import YOLO
from io import BytesIO


def estimate_yield():
    output_csv = './output/yield_results2.csv'
    model = YOLO('../runs/detect/yield_estimation_project/orange_detection_v12/weights/best.pt')
    results_data = []

    st.write("## Estimated Yield")

    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files='directory'
    )

    if uploaded_files:

        tab1, tab2 = st.tabs(["Images", "Estimated Yield"])

        for uploaded_file in uploaded_files:
            # Open image with PIL
            #image = Image.open(uploaded_file[0])
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            # Convert to bytes for YOLO
            #image_bytes = uploaded_file.read()

            # Run prediction directly on image
            results = model.predict(image, conf=0.1, save=False, verbose=False)

            count = len(results[0].boxes)

            results_data.append({
                'image_name': uploaded_file.name,
                'estimated_yield': count
            })

            tab1.subheader(uploaded_file.name)
            tab1.image(image, use_container_width=True)

        # Save CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_name', 'estimated_yield'])
            writer.writeheader()
            writer.writerows(results_data)

        tab2.subheader("Estimated Yield Results")
        tab2.write(results_data)


if __name__ == "__main__":
    estimate_yield()
