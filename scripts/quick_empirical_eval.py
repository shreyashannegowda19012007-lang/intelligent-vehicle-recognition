"""
Quick Empirical Evaluation Script:
Compares classification accuracy, top-5 accuracy, make accuracy,
average confidence, and top-1 vs top-2 probability margins between:
1. Legacy Pipeline (224x224 letterbox downsampling + raw DL)
2. Improved Hybrid Pipeline (Native resolution crop + Principled Decision Fusion)
Across the known BMW M5 test image and a representative sample of 40 validation images.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, ".")
from src.classification.pipeline import VehicleRecognitionPipeline
from src.hybrid.fusion_pipeline import HybridVehiclePipeline
from src.classical_cv.roi_extractor import ROIExtractor

def main():
    print("=" * 70)
    print("QUICK EMPIRICAL BENCHMARK: PREPROCESSING & HYBRID FUSION COMPARISON")
    print("=" * 70)

    pipeline = VehicleRecognitionPipeline()
    hybrid = HybridVehiclePipeline()
    legacy_roi_extractor = ROIExtractor(target_size=(224, 224), keep_aspect_ratio=True, padding_ratio=0.05)

    # -------------------------------------------------------------
    # 1. SPECIFIC TEST ON KNOWN BMW M5 (00061.jpg)
    # -------------------------------------------------------------
    bmw_path = "Stanford Dataset/cars_test/cars_test/00061.jpg"
    print(f"\n[1] Evaluating Specific BMW M5 Test Image: {bmw_path}")
    bmw_img = Image.open(bmw_path)
    
    # Detect bbox
    det_res = pipeline.detector.detect(bmw_img, extract_rois=False)
    dets = det_res.get("detections", [])
    if dets:
        b = dets[0]["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
    else:
        x1, y1, x2, y2 = 0, 0, bmw_img.width, bmw_img.height

    # Legacy: 224x224 letterbox array
    legacy_roi, _ = legacy_roi_extractor.extract_roi(np.array(bmw_img), {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    legacy_pred = pipeline.classifier.predict(legacy_roi)
    
    # Improved: Native crop
    native_crop = bmw_img.crop((max(0, x1), max(0, y1), min(bmw_img.width, x2), min(bmw_img.height, y2)))
    improved_deep = pipeline.classifier.predict(native_crop)
    improved_hybrid = hybrid.predict(native_crop)

    print(f"  * Legacy 224px Letterbox Pred:  {legacy_pred['predicted_class']} (Conf: {legacy_pred['confidence']*100:.1f}%)")
    print(f"  * Improved Native DL Pred:       {improved_deep['predicted_class']} (Conf: {improved_deep['confidence']*100:.1f}%)")
    print(f"  * Improved Hybrid Decision Pred: {improved_hybrid['predicted_class']} (Conf: {improved_hybrid['confidence']*100:.1f}%)")
    print(f"  * Ground Truth:                  BMW M5 Sedan 2010 (Make: BMW)")
    print(f"  * Verdict: Legacy = {'CORRECT' if 'BMW M5' in legacy_pred['predicted_class'] else 'WRONG'}, Improved = {'CORRECT' if 'BMW M5' in improved_hybrid['predicted_class'] else 'WRONG'}")

    # -------------------------------------------------------------
    # 2. REPRESENTATIVE SAMPLE OF 40 VALIDATION IMAGES
    # -------------------------------------------------------------
    val_csv = "data/splits/validation.csv"
    df_val = pd.read_csv(val_csv)
    # Balanced random sample across validation split
    sample_df = df_val.sample(40, random_state=42)
    print(f"\n[2] Evaluating Representative Sample of {len(sample_df)} Validation Images with Ground Truth...")

    results = []

    for idx, row in sample_df.iterrows():
        img_p = row["image_path"]
        if not Path(img_p).is_file():
            continue
        
        gt_class = str(row["class_name"]).strip()
        gt_make = str(row["make"]).strip()
        gt_model = str(row["model"]).strip()
        
        # Ground truth bbox
        x1 = max(0, int(row["bbox_x1"]))
        y1 = max(0, int(row["bbox_y1"]))
        x2 = max(x1 + 1, int(row["bbox_x2"]))
        y2 = max(y1 + 1, int(row["bbox_y2"]))

        with Image.open(img_p) as raw_img:
            img = raw_img.convert("RGB")
            
            # --- Condition A: Legacy (224x224 letterboxed) ---
            leg_roi, _ = legacy_roi_extractor.extract_roi(np.array(img), {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            leg_p = pipeline.classifier.predict(leg_roi)
            
            # --- Condition B: Improved Native DL (320x320 clean crop) ---
            crop = img.crop((x1, y1, min(img.width, x2), min(img.height, y2)))
            dl_p = pipeline.classifier.predict(crop)
            
            # --- Condition C: Improved Hybrid System ---
            hyb_p = hybrid.predict(crop)

        # Check accuracy
        def eval_pred(pred):
            top1_c = pred.get("predicted_class", "")
            top1_acc = (top1_c == gt_class)
            make_acc = (str(pred.get("make", "")).strip().lower() == gt_make.lower())
            top5_classes = [t.get("class", "") for t in pred.get("top_5", [])]
            top5_acc = (gt_class in top5_classes)
            conf = float(pred.get("confidence", 0.0))
            
            # Margin: Top-1 vs Top-2
            if len(pred.get("top_5", [])) >= 2:
                margin = conf - float(pred["top_5"][1].get("confidence", 0.0))
            else:
                margin = conf

            return {
                "top1_acc": top1_acc,
                "top5_acc": top5_acc,
                "make_acc": make_acc,
                "conf": conf,
                "margin": margin,
            }

        res_leg = eval_pred(leg_p)
        res_dl = eval_pred(dl_p)
        res_hyb = eval_pred(hyb_p)

        results.append({
            "legacy": res_leg,
            "improved_dl": res_dl,
            "improved_hybrid": res_hyb,
        })

    n = len(results)
    print(f"\nCompleted evaluation on {n} images.")

    def aggregate(key):
        acc1 = np.mean([r[key]["top1_acc"] for r in results]) * 100.0
        acc5 = np.mean([r[key]["top5_acc"] for r in results]) * 100.0
        make_acc = np.mean([r[key]["make_acc"] for r in results]) * 100.0
        avg_conf = np.mean([r[key]["conf"] for r in results]) * 100.0
        avg_margin = np.mean([r[key]["margin"] for r in results]) * 100.0
        return acc1, acc5, make_acc, avg_conf, avg_margin

    leg_a1, leg_a5, leg_mk, leg_cf, leg_mg = aggregate("legacy")
    dl_a1, dl_a5, dl_mk, dl_cf, dl_mg = aggregate("improved_dl")
    hyb_a1, hyb_a5, hyb_mk, hyb_cf, hyb_mg = aggregate("improved_hybrid")

    print("\n" + "=" * 70)
    print("EMPIRICAL COMPARISON RESULTS TABLE")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Legacy (224px)':<15} | {'Native DL (320px)':<17} | {'Proposed Hybrid':<15}")
    print("-" * 85)
    print(f"{'Top-1 Exact Class Accuracy':<30} | {leg_a1:>13.2f}% | {dl_a1:>15.2f}% | {hyb_a1:>13.2f}%")
    print(f"{'Top-5 Class Accuracy':<30} | {leg_a5:>13.2f}% | {dl_a5:>15.2f}% | {hyb_a5:>13.2f}%")
    print(f"{'Manufacturer (Make) Accuracy':<30} | {leg_mk:>13.2f}% | {dl_mk:>15.2f}% | {hyb_mk:>13.2f}%")
    print(f"{'Mean Prediction Confidence':<30} | {leg_cf:>13.2f}% | {dl_cf:>15.2f}% | {hyb_cf:>13.2f}%")
    print(f"{'Mean Top-1 vs Top-2 Margin':<30} | {leg_mg:>13.2f}% | {dl_mg:>15.2f}% | {hyb_mg:>13.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
