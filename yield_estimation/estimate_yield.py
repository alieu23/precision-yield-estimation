import os
import csv
from ultralytics import YOLO

# Verify this path matches your latest training run
model_path = '../runs/detect/yield_estimation_project/orange_detection_v12/weights/best.pt'
model = YOLO(model_path)

input_folder = './orange_images'
output_csv = './output/yield_results.csv'
# Define where the images with boxes will be saved
output_visuals_dir = './output/visuals'


def run_estimate():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    results_data = []

    images = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Start yield estimation on {len(images)} images..")

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)

        # UPDATED: save=True draws boxes and confidence scores
        # project/name tells YOLO where to put the results
        results = model.predict(
            source=img_path,
            conf=0.1,
            save=True,  # Draws bounding boxes
            save_conf=True,  # Specifically ensures confidence is visible
            project='./output',  # Parent folder
            name='visuals',  # Sub-folder
            exist_ok=True,  # Overwrites/merges into the same folder
            verbose=False
        )

        # count detections, one class
        count = len(results[0].boxes)

        results_data.append({'image_name': img_name, 'estimated_yield': count})
        print(f"Processed {img_name}: {count} oranges found")

    # Export result to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'estimated_yield'])
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\nDone! Report saved to {output_csv}")
    print(f"Visual detections saved in: ./output/visuals")


if __name__ == '__main__':
    run_estimate()