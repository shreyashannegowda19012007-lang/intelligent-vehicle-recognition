"""
CLI Script for Evaluating Trained Classical Computer Vision Models.
Usage:
    python scripts/evaluate_classical.py --model-path models/classical/svm/linear_svm_hog_color_lbp_roi_model.joblib --split val
    python scripts/evaluate_classical.py --model-path models/classical/random_forest/random_forest_hog_roi_model.joblib --split test --save-predictions outputs/classical_cv/predictions/test_preds.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.classical_cv.evaluator import ClassicalEvaluator
from src.classical_cv.predictor import ClassicalVehiclePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_classical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classical CV model on validation or test split")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained classifier .joblib artifact",
    )
    parser.add_argument(
        "--pipeline-path",
        type=str,
        default=None,
        help="Path to fitted FeatureFusionPipeline .joblib artifact",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "validation", "test", "train"],
        help="Split to evaluate on",
    )
    parser.add_argument(
        "--split-csv",
        type=str,
        default=None,
        help="Custom split CSV path",
    )
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="Disable bounding box ROI crop",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to evaluate",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        help="File path to save prediction results CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.is_file():
        logger.error("Model file not found: %s", model_path)
        return 1

    # Resolve split CSV
    if args.split_csv:
        split_p = Path(args.split_csv)
    else:
        split_name = "validation.csv" if args.split in ("val", "validation") else f"{args.split}.csv"
        split_p = PROJECT_ROOT / "data" / "splits" / split_name

    if not split_p.is_file():
        logger.error("Split file not found: %s", split_p)
        return 1

    df = pd.read_csv(split_p)
    logger.info("Loaded %d samples from %s", len(df), split_p)

    predictor = ClassicalVehiclePredictor(
        model_path=model_path,
        pipeline_path=args.pipeline_path,
        project_root=PROJECT_ROOT,
    )

    is_labeled = "class_id" in df.columns and df["class_id"].notna().any() and (df["class_id"] >= 0).any()

    if is_labeled:
        logger.info("Labeled split detected; extracting features for full evaluation...")
        pipeline = predictor.pipeline
        use_bbox = not args.no_bbox

        X_raw, y_true, img_ids = pipeline.extract_from_dataframe(
            df=df,
            project_root=PROJECT_ROOT,
            use_bbox=use_bbox,
            max_samples=args.max_samples,
        )
        X = pipeline.transform(X_raw)

        evaluator = ClassicalEvaluator(project_root=PROJECT_ROOT)
        metrics = evaluator.evaluate(predictor.classifier, X, y_true, image_ids=img_ids)

        print("\n" + "=" * 60)
        print(f"EVALUATION METRICS ({split_p.name})")
        print("=" * 60)
        print(f"Evaluated Samples:       {metrics['num_samples']}")
        print(f"Top-1 Accuracy:          {metrics['top1_accuracy'] * 100:.2f}%")
        print(f"Top-5 Accuracy:          {metrics['top5_accuracy'] * 100:.2f}%")
        print(f"Precision (Macro):       {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro):          {metrics['recall_macro']:.4f}")
        print(f"F1 Score (Macro):        {metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted):     {metrics['f1_weighted']:.4f}")
        print(f"Inference Latency:       {metrics['latency_per_sample_ms']:.2f} ms / sample")
        print("=" * 60 + "\n")

    if args.save_predictions or not is_labeled:
        logger.info("Generating prediction records...")
        preds_df = predictor.predict_dataframe(
            df=df,
            use_bbox=not args.no_bbox,
            max_samples=args.max_samples,
        )
        out_pred_path = Path(args.save_predictions) if args.save_predictions else PROJECT_ROOT / "outputs" / "classical_cv" / "predictions" / f"{model_path.stem}_{split_p.stem}_predictions.csv"
        out_pred_path.parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(out_pred_path, index=False)
        logger.info("Saved %d predictions to %s", len(preds_df), out_pred_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
