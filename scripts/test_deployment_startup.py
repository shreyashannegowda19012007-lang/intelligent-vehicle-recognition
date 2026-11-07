import os
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.checkpoint import ensure_convnext_checkpoint
from src.classification.pipeline import VehicleRecognitionPipeline
from src.hybrid.fusion_pipeline import HybridVehiclePipeline

print("1. Running checkpoint reassembly check...")
ckpt_path = ensure_convnext_checkpoint(PROJECT_ROOT)
print(f"Checkpoint ready at: {ckpt_path} ({ckpt_path.stat().st_size / (1024*1024):.2f} MB)")

print("\n2. Initializing CPU-only pipelines as Streamlit Community Cloud will do...")
pipe = VehicleRecognitionPipeline(device="cpu")
hybrid = HybridVehiclePipeline(device="cpu")

print(f"Classical predictor available: {hybrid.classical_predictor is not None}")
assert hybrid.classical_predictor is not None, "Classical predictor should be loaded!"

print("\n3. Running end-to-end CPU inference on test image 00061.jpg...")
img_path = PROJECT_ROOT / "Stanford Dataset" / "cars_test" / "cars_test" / "00061.jpg"
img = Image.open(img_path).convert("RGB")

det_res = pipe.process_image(img)
print(f"Detections found: {det_res['num_vehicles_detected']}")
top_det = det_res["detections"][0]["classification"]
print(f"Detection pipeline Top-1: {top_det['predicted_class']} ({top_det['confidence']*100:.2f}%)")

hyb_res = hybrid.predict(img)
print(f"Hybrid pipeline Top-1: {hyb_res['predicted_class']} ({hyb_res['confidence']*100:.2f}%)")
print(f"Consensus agreement: {hyb_res.get('agreement', 'N/A')}")

print("\nALL DEPLOYMENT REASSEMBLY & INFERENCE CHECKS PASSED!")

