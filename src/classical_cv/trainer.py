"""
Unified Classical Computer Vision Training Pipeline.
Loads splits, handles feature extraction with disk caching, fits feature normalization
(StandardScaler & PCA) strictly on training data, trains SVM or Random Forest classifiers,
evaluates on validation data, and persists model artifacts and metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

from src.classical_cv.evaluator import ClassicalEvaluator
from src.classical_cv.feature_fusion import FeatureFusionPipeline
from src.classical_cv.roi_extractor import ROIExtractor
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class ClassicalTrainer:
    """
    Orchestrates classical computer vision model training and evaluation.
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.root = Path(project_root).resolve() if project_root else find_project_root()
        cfg_file = Path(config_path) if config_path else self.root / "configs" / "config.yaml"
        if cfg_file.is_file():
            try:
                with open(cfg_file, encoding="utf-8") as f:
                    self.raw_cfg = yaml.safe_load(f) or {}
            except Exception:  # noqa: BLE001
                self.raw_cfg = {}
        else:
            self.raw_cfg = {}

        self.cv_cfg = self.raw_cfg.get("classical_cv", {})
        self.evaluator = ClassicalEvaluator(project_root=self.root)

    def _build_classifier(
        self,
        classifier_type: str = "linear_svm",
        svm_c: float = 1.0,
        svm_kernel: str = "linear",
        calibrate_svm: bool = True,
        rf_n_estimators: int = 100,
        rf_max_depth: int | None = 25,
        rf_min_samples_leaf: int = 2,
        random_state: int = 42,
        min_class_samples: int | None = None,
    ) -> Any:
        """Instantiate scikit-learn classifier based on configuration."""
        c_type = classifier_type.lower()

        if c_type in ("linear_svm", "svm") and svm_kernel == "linear":
            base_svm = LinearSVC(C=svm_c, dual=False, tol=1e-3, max_iter=1000, random_state=random_state)
            if calibrate_svm:
                if min_class_samples is not None and min_class_samples < 2:
                    logger.warning("Class has < 2 samples in training subset; using uncalibrated LinearSVC")
                    return base_svm
                cv_folds = 3 if (min_class_samples is None or min_class_samples >= 3) else 2
                return CalibratedClassifierCV(estimator=base_svm, cv=cv_folds)
            return base_svm

        if c_type in ("rbf_svm", "svm") and svm_kernel == "rbf":
            return SVC(C=svm_c, kernel="rbf", probability=True, random_state=random_state)

        if c_type in ("random_forest", "rf"):
            return RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_leaf=rf_min_samples_leaf,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            )

        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    def train_and_evaluate(
        self,
        feature_config: str = "hog_color_lbp",
        classifier_type: str = "linear_svm",
        use_bbox: bool = True,
        use_scaler: bool = True,
        use_pca: bool = False,
        pca_components: float | None = 128,
        svm_c: float = 1.0,
        svm_kernel: str = "linear",
        calibrate_svm: bool = True,
        rf_n_estimators: int = 100,
        rf_max_depth: int | None = 25,
        rf_min_samples_leaf: int = 2,
        train_csv: str | Path | None = None,
        val_csv: str | Path | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        random_state: int = 42,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute end-to-end training and validation.

        Returns:
            Dictionary containing metrics, artifact paths, and configuration.
        """
        # 1. Resolve manifests
        train_p = Path(train_csv) if train_csv else self.root / "data" / "splits" / "train.csv"
        val_p = Path(val_csv) if val_csv else self.root / "data" / "splits" / "validation.csv"

        df_train = pd.read_csv(train_p)
        df_val = pd.read_csv(val_p)

        logger.info("Loaded splits: %d train, %d val samples", len(df_train), len(df_val))

        # 2. Setup pipeline
        roi_ext = ROIExtractor(
            target_size=(128, 128),
            keep_aspect_ratio=True,
            padding_ratio=0.05,
        )
        pipeline = FeatureFusionPipeline(
            feature_config=feature_config,
            roi_extractor=roi_ext,
            use_scaler=use_scaler,
            use_pca=use_pca,
            pca_components=pca_components,
            random_state=random_state,
        )

        # 3. Extract raw features with caching
        cache_dir = self.root / "data" / "cache" / "classical_features"
        bbox_tag = "roi" if use_bbox else "full"
        train_limit_tag = f"_n{max_train_samples}" if max_train_samples else "_all"
        val_limit_tag = f"_n{max_val_samples}" if max_val_samples else "_all"

        train_cache = cache_dir / f"train_{feature_config}_{bbox_tag}{train_limit_tag}.npz"
        val_cache = cache_dir / f"val_{feature_config}_{bbox_tag}{val_limit_tag}.npz"

        logger.info("Extracting / loading training features...")
        has_orb = ("orb" in feature_config or feature_config == "all")
        X_train_raw, y_train, _train_ids = pipeline.extract_from_dataframe(
            df=df_train,
            project_root=self.root,
            cache_path=train_cache,
            use_bbox=use_bbox,
            max_samples=max_train_samples,
            fit_orb_vocab=has_orb,
        )

        logger.info("Extracting / loading validation features...")
        X_val_raw, y_val, val_ids = pipeline.extract_from_dataframe(
            df=df_val,
            project_root=self.root,
            cache_path=val_cache,
            use_bbox=use_bbox,
            max_samples=max_val_samples,
            fit_orb_vocab=False,
        )

        # 4. Fit feature transformations strictly on training features
        logger.info("Fitting scaler and PCA on training data only...")
        X_train = pipeline.fit_transform(X_train_raw)
        X_val = pipeline.transform(X_val_raw)

        # 5. Build and train classifier
        logger.info("Training %s classifier on %d samples (dim=%d)...", classifier_type, len(X_train), X_train.shape[1])
        class_counts = pd.Series(y_train).value_counts()
        min_class_samples = int(class_counts.min()) if not class_counts.empty else 0
        clf = self._build_classifier(
            classifier_type=classifier_type,
            svm_c=svm_c,
            svm_kernel=svm_kernel,
            calibrate_svm=calibrate_svm,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            rf_min_samples_leaf=rf_min_samples_leaf,
            random_state=random_state,
            min_class_samples=min_class_samples,
        )
        clf.fit(X_train, y_train)

        # 6. Evaluate on validation set
        logger.info("Evaluating on %d validation samples...", len(X_val))
        val_metrics = self.evaluator.evaluate(clf, X_val, y_val, image_ids=val_ids)
        failures = self.evaluator.analyze_failures(clf, X_val, y_val, image_ids=val_ids, max_samples=25)

        # 7. Persist artifacts
        tag = run_name or f"{classifier_type}_{feature_config}_{bbox_tag}"
        sub_dir = "svm" if "svm" in classifier_type.lower() else "random_forest"

        model_dir = self.root / "models" / "classical" / sub_dir
        scaler_dir = self.root / "models" / "classical" / "scalers"
        vocab_dir = self.root / "models" / "classical" / "vocabularies"
        metrics_dir = self.root / "outputs" / "classical_cv" / "metrics"
        pred_dir = self.root / "outputs" / "classical_cv" / "predictions"

        model_dir.mkdir(parents=True, exist_ok=True)
        scaler_dir.mkdir(parents=True, exist_ok=True)
        vocab_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        model_file = model_dir / f"{tag}_model.joblib"
        scaler_file = scaler_dir / f"{tag}_pipeline.joblib"
        metrics_file = metrics_dir / f"{tag}_metrics.json"
        failures_file = pred_dir / f"{tag}_failures.json"
        vocab_file = vocab_dir / f"{tag}_vocab.joblib" if pipeline.orb_extractor.kmeans is not None else None

        # Save model package
        joblib.dump({
            "classifier": clf,
            "pipeline": pipeline,
            "metadata": {
                "tag": tag,
                "feature_config": feature_config,
                "classifier_type": classifier_type,
                "use_bbox": use_bbox,
                "num_train_samples": len(X_train),
                "num_features": X_train.shape[1],
                "val_top1": val_metrics["top1_accuracy"],
                "val_top5": val_metrics["top5_accuracy"],
            },
        }, model_file)

        pipeline.save_pipeline(scaler_file)
        if vocab_file is not None:
            pipeline.orb_extractor.save_vocabulary(vocab_file)
        self.evaluator.save_metrics(val_metrics, metrics_file)
        self.evaluator.save_failures(failures, failures_file)

        logger.info(
            "Completed %s. Val Top-1: %.2f%%, Val Top-5: %.2f%%, Latency: %.2f ms/sample",
            tag,
            val_metrics["top1_accuracy"] * 100,
            val_metrics["top5_accuracy"] * 100,
            val_metrics["latency_per_sample_ms"],
        )

        return {
            "run_name": tag,
            "classifier_type": classifier_type,
            "feature_config": feature_config,
            "use_bbox": use_bbox,
            "num_train": len(X_train),
            "num_val": len(X_val),
            "feature_dim": X_train.shape[1],
            "metrics": val_metrics,
            "failures": failures,
            "model_path": str(model_file),
            "scaler_path": str(scaler_file),
            "vocab_path": str(vocab_file) if vocab_file else None,
            "metrics_path": str(metrics_file),
            "failures_path": str(failures_file),
        }
