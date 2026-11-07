import sys
sys.path.insert(0, '.')
from PIL import Image
from src.classification.pipeline import VehicleRecognitionPipeline
from src.hybrid.fusion_pipeline import HybridVehiclePipeline

print('Testing CPU-only initialization...')
pipe_cpu = VehicleRecognitionPipeline(device='cpu')
hyb_cpu = HybridVehiclePipeline(device='cpu')

img = Image.open('Stanford Dataset/cars_test/cars_test/00061.jpg')
print('Running detector on CPU...')
det_res = pipe_cpu.detector.detect(img, extract_rois=False)
print('Detections on CPU:', len(det_res['detections']))

print('Running classifier on CPU...')
cls_res = pipe_cpu.classifier.predict(img)
print(f"Classifier pred on CPU: {cls_res['predicted_class']} ({cls_res['confidence']*100:.1f}%)")

print('Running hybrid on CPU...')
hyb_res = hyb_cpu.predict(img)
print(f"Hybrid pred on CPU: {hyb_res['predicted_class']} ({hyb_res['confidence']*100:.1f}%)")
print('CPU inference successfully validated!')
