"""
CLI Script for Training Classical Computer Vision Models.
Usage:
    python scripts/train_classical.py --feature-config hog_color_lbp --classifier linear_svm
    python scripts/train_classical.py --feature-config hog --classifier random_forest --max-train-samples 1000
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

from src.classical_cv.trainer import ClassicalTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_classical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical CV baseline on Stanford Cars")
    parser.add_argument(
        "--feature-config",
        type=str,
        default="hog_color_lbp",
        choices=["hog", "color", "lbp", "orb", "hog_color", "hog_lbp", "hog_orb", "hog_color_lbp", "all", "hog_color_lbp_orb"],
        help="Feature descriptor configuration to extract and fuse",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="linear_svm",
        choices=["linear_svm", "rbf_svm", "random_forest"],
        help="Classical classifier architecture",
    )
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="If set, extracts features from full image without cropping vehicle ROI",
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="If set, applies PCA dimensionality reduction",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=128,
        help="Number of PCA components if --use-pca is active",
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=1.0,
        help="Regularization parameter C for SVM",
    )
    parser.add_argument(
        "--rf-trees",
        type=int,
        default=100,
        help="Number of decision trees for Random Forest",
    )
    parser.add_argument(
        "--rf-depth",
        type=int,
        default=25,
        help="Max depth for Random Forest trees",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit number of training images for fast subset experiments",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Limit number of validation images for fast subset experiments",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name for artifact persistence",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("Starting Classical CV Training with config: feature=%s, classifier=%s", args.feature_config, args.classifier)

    trainer = ClassicalTrainer(project_root=PROJECT_ROOT)
    result = trainer.train_and_evaluate(
        feature_config=args.feature_config,
        classifier_type=args.classifier,
        use_bbox=not args.no_bbox,
        use_scaler=True,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        svm_c=args.svm_c,
        rf_n_estimators=args.rf_trees,
        rf_max_depth=args.rf_depth,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        run_name=args.run_name,
    )

    metrics = result["metrics"]
    print("\n" + "=" * 60)
    print(f"EXPERIMENT RESULTS: {result['run_name']}")
    print("=" * 60)
    print(f"Training Samples:        {result['num_train']}")
    print(f"Validation Samples:      {result['num_val']}")
    print(f"Feature Dimension:       {result['feature_dim']}")
    print(f"Validation Top-1 Acc:    {metrics['top1_accuracy'] * 100:.2f}%")
    print(f"Validation Top-5 Acc:    {metrics['top5_accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score:          {metrics['f1_macro']:.4f}")
    print(f"Weighted F1 Score:       {metrics['f1_weighted']:.4f}")
    print(f"Inference Latency:       {metrics['latency_per_sample_ms']:.2f} ms / sample")
    print(f"Model Artifact:          {result['model_path']}")
    print(f"Metrics File:            {result['metrics_path']}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
