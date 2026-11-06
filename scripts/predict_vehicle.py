"""
Vehicle Prediction CLI.
Runs vehicle detection, outputs bounding boxes, saves visual overlays and ROI crops.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.inference import VehicleClassifier
from src.detection.detector import VehicleDetector
from src.detection.visualizer import VehicleVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict_vehicle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict vehicle bounding boxes and extract ROIs")
    parser.add_argument("--image", type=str, required=True, help="Image file path or directory of images")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO checkpoint (default: models/detection/best.pt or yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    parser.add_argument("--device", type=str, default="auto", help="Inference device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--save-vis", type=str, default="outputs/detection/visualizations", help="Directory to save visual overlays")
    parser.add_argument("--save-rois", type=str, default="outputs/detection/visualizations/rois", help="Directory to save ROI crops")
    parser.add_argument("--classifier", type=str, default=None, help="Optional path to classification checkpoint (e.g. outputs/classification/checkpoints/best_checkpoint.pt)")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = PROJECT_ROOT / args.image if not Path(args.image).is_file() and not Path(args.image).is_dir() else Path(args.image)

    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    else:
        logger.error("Input path does not exist: %s", input_path)
        return 1

    detector = VehicleDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )
    visualizer = VehicleVisualizer()
    classifier = None
    if args.classifier:
        classifier = VehicleClassifier(
            checkpoint_path=args.classifier,
            device=args.device,
        )

    vis_dir = PROJECT_ROOT / args.save_vis
    roi_dir = PROJECT_ROOT / args.save_rois
    vis_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for img_p in image_files:
        logger.info("Processing %s...", img_p.name)
        pred = detector.detect(img_p, extract_rois=True)

        # Generate visual overlay
        annotated = visualizer.draw_detections(img_p, pred["detections"])
        vis_save_path = vis_dir / f"pred_{img_p.name}"
        visualizer.save_image(annotated, vis_save_path)

        # Save ROI crops and classify
        for idx, det in enumerate(pred["detections"]):
            if det["roi"] is not None:
                roi_save_path = roi_dir / f"{img_p.stem}_roi_{idx+1}.jpg"
                visualizer.save_image(det["roi"], roi_save_path)
                if classifier is not None:
                    cls_out = classifier.predict(det["roi"])
                    det["classification"] = cls_out
            elif classifier is not None:
                # If no ROI, classify full image
                det["classification"] = classifier.predict(img_p)

        # Serialized output dict (remove numpy array for JSON)
        serializable_dets = []
        for det in pred["detections"]:
            det_copy = dict(det)
            det_copy["roi"] = f"<ndarray shape={det['roi'].shape}>" if det.get("roi") is not None else None
            serializable_dets.append(det_copy)

        res_entry = {
            "image": str(img_p.relative_to(PROJECT_ROOT) if img_p.is_relative_to(PROJECT_ROOT) else img_p).replace("\\", "/"),
            "detections": serializable_dets,
            "inference_time_ms": pred["inference_time_ms"],
            "image_width": pred["image_width"],
            "image_height": pred["image_height"],
            "visualization_path": str(vis_save_path.relative_to(PROJECT_ROOT) if vis_save_path.is_relative_to(PROJECT_ROOT) else vis_save_path).replace("\\", "/"),
        }
        all_results.append(res_entry)

    if args.output_json:
        out_json_p = PROJECT_ROOT / args.output_json
        out_json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_p, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Saved prediction results to %s", out_json_p)
    else:
        print(json.dumps(all_results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
