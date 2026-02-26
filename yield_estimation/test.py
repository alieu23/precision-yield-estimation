from ultralytics import YOLO
import os
import csv


output_csv = './output/evaluation_metrics.csv'
results= []

model = YOLO('../runs/detect/yield_estimation_project/orange_detection_v12/weights/best.pt')

metrics = model.val(split='test')
results.append({'mAP50': metrics.box.map50, 'Precision': metrics.box.mp, 'Recall': metrics.box.mr})
with open (output_csv,'w', newline='') as f:
    w = csv.DictWriter(f,fieldnames =['mAP50', 'Precision', 'Recall'])
    w.writeheader()
    w.writerows(results)


print(f"Final Test mAP50: {metrics.box.map50}")
print(f"Final Test Precision: {metrics.box.mp}")
print(f"Final Test Recall: {metrics.box.mr}")