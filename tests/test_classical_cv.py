"""
Automated Unit and Integration Tests for Classical Computer Vision Baseline.
Validates:
1. ROI extraction
2. HOG feature dimensions
3. Colour histogram dimensions
4. LBP extraction
5. ORB extraction
6. Feature fusion
7. Feature normalization
8. Classifier training
9. Top-5 predictions
10. Class mapping
11. Artifact saving and loading
12. Prediction output format
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.classical_cv.color_features import ColorHistogramExtractor
from src.classical_cv.evaluator import ClassicalEvaluator
from src.classical_cv.feature_fusion import FeatureFusionPipeline
from src.classical_cv.hog_features import HOGFeatureExtractor
from src.classical_cv.orb_features import ORBFeatureExtractor
from src.classical_cv.predictor import ClassicalVehiclePredictor
from src.classical_cv.roi_extractor import ROIExtractor
from src.classical_cv.texture_features import LBPFeatureExtractor
from src.classical_cv.trainer import ClassicalTrainer
from src.utils.config import find_project_root


@pytest.fixture(scope="session")
def project_root() -> Path:
    return find_project_root()


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """Provide a deterministic synthetic RGB vehicle-like image."""
    rng = np.random.default_rng(42)
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    # Add simple geometric vehicle structure
    img[50:180, 40:260, :] = 180  # body
    img[70:120, 80:220, :] = 80   # cabin/windows
    img[150:185, 60:100, :] = 30  # front wheel
    img[150:185, 200:240, :] = 30 # rear wheel
    noise = rng.integers(0, 20, size=img.shape, dtype=np.uint8)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


# 1. ROI Extraction
def test_roi_extraction(synthetic_image: np.ndarray):
    extractor = ROIExtractor(target_size=(128, 128), keep_aspect_ratio=True, padding_ratio=0.05)

    # Valid bounding box
    bbox = (40, 50, 260, 180)
    crop, meta = extractor.extract_roi(synthetic_image, bbox=bbox, use_bbox=True)
    assert crop.shape == (128, 128, 3)
    assert meta["fallback_used"] is False
    assert meta["target_size"] == (128, 128)

    # Padding check: applied bbox should be wider/taller than parsed bbox
    px1, py1, px2, py2 = meta["applied_bbox"]
    assert px1 <= 40
    assert py1 <= 50
    assert px2 >= 260
    assert py2 >= 180

    # Invalid bounding box fallback
    invalid_box = (200, 200, 50, 50)
    crop_fallback, meta_fallback = extractor.extract_roi(synthetic_image, bbox=invalid_box, use_bbox=True)
    assert crop_fallback.shape == (128, 128, 3)
    assert meta_fallback["fallback_used"] is True

    # Full frame extraction (use_bbox=False)
    crop_full, meta_full = extractor.extract_roi(synthetic_image, bbox=bbox, use_bbox=False)
    assert crop_full.shape == (128, 128, 3)
    assert meta_full["fallback_used"] is False


# 2. HOG Feature Dimensions
def test_hog_feature_dimensions(synthetic_image: np.ndarray):
    hog_ext = HOGFeatureExtractor(
        image_size=(128, 128),
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
    )
    dim = hog_ext.get_feature_dimension()
    assert dim == 8100

    features = hog_ext.extract_features(synthetic_image)
    assert features.shape == (8100,)
    assert features.dtype == np.float32

    # Check visualization output
    _, viz = hog_ext.compute_hog_with_visualization(synthetic_image)
    assert viz.shape == (128, 128)
    assert viz.dtype == np.uint8


# 3. Colour Histogram Dimensions
def test_color_histogram_dimensions(synthetic_image: np.ndarray):
    color_ext = ColorHistogramExtractor(
        color_spaces=["rgb", "hsv", "lab"],
        bins_per_channel=16,
        norm="l2",
    )
    # 3 spaces * 3 channels * 16 bins = 144
    dim = color_ext.get_feature_dimension()
    assert dim == 144

    features = color_ext.extract_features(synthetic_image)
    assert features.shape == (144,)
    # Verify L2 normalization: norm should be close to 1.0
    norm_val = float(np.linalg.norm(features))
    assert abs(norm_val - 1.0) < 1e-4

    hists = color_ext.compute_channel_histograms_dict(synthetic_image)
    assert "rgb" in hists
    assert "hsv" in hists
    assert "lab" in hists


# 4. LBP Extraction
def test_lbp_extraction(synthetic_image: np.ndarray):
    lbp_ext = LBPFeatureExtractor(
        radius=2,
        n_points=16,
        method="uniform",
        grid=(4, 4),
        norm="l2",
        image_size=(128, 128),
    )
    # 4 * 4 * (16 + 2) = 288
    dim = lbp_ext.get_feature_dimension()
    assert dim == 288

    features = lbp_ext.extract_features(synthetic_image)
    assert features.shape == (288,)
    assert features.dtype == np.float32

    _, viz = lbp_ext.compute_lbp_with_visualization(synthetic_image)
    assert viz.shape == (128, 128)


# 5. ORB Extraction
def test_orb_extraction(synthetic_image: np.ndarray):
    orb_ext = ORBFeatureExtractor(vocab_size=32, n_features=200)

    # Keypoints and descriptors
    kps, des = orb_ext.detect_and_compute(synthetic_image)
    assert len(kps) > 0
    assert des is not None
    assert des.shape[1] == 32

    # Descriptor statistics
    stats = orb_ext.get_descriptor_statistics(synthetic_image)
    assert stats["keypoint_count"] > 0
    assert "mean_response" in stats

    # Fit visual vocabulary
    train_imgs = [synthetic_image, np.fliplr(synthetic_image)]
    orb_ext.fit_vocabulary(train_imgs, max_descriptors=1000)
    assert orb_ext.kmeans is not None

    # Visual words histogram
    bovw = orb_ext.extract_features(synthetic_image)
    assert bovw.shape == (32,)
    assert bovw.dtype == np.float32

    # Keypoint drawing
    viz = orb_ext.draw_keypoints(synthetic_image, max_keypoints=50)
    assert viz.shape == synthetic_image.shape


# 6. Feature Fusion
def test_feature_fusion(synthetic_image: np.ndarray):
    pipeline = FeatureFusionPipeline(feature_config="hog_color_lbp")
    raw = pipeline.extract_raw_descriptors(synthetic_image)

    # 8100 (HOG) + 96 (Color HSV+RGB 16 bins) + 288 (LBP) = 8484
    expected_dim = 8100 + (2 * 3 * 16) + 288
    assert raw.shape == (expected_dim,)
    assert raw.dtype == np.float32

    # Individual configs
    assert FeatureFusionPipeline(feature_config="hog").extract_raw_descriptors(synthetic_image).shape == (8100,)
    assert FeatureFusionPipeline(feature_config="color").extract_raw_descriptors(synthetic_image).shape == (96,)
    assert FeatureFusionPipeline(feature_config="lbp").extract_raw_descriptors(synthetic_image).shape == (288,)


# 7. Feature Normalization
def test_feature_normalization():
    rng = np.random.default_rng(42)
    X_train = rng.normal(loc=10.0, scale=5.0, size=(50, 100)).astype(np.float32)
    X_val = rng.normal(loc=10.0, scale=5.0, size=(20, 100)).astype(np.float32)

    pipeline = FeatureFusionPipeline(use_scaler=True, use_pca=True, pca_components=16)
    X_train_trans = pipeline.fit_transform(X_train)
    X_val_trans = pipeline.transform(X_val)

    assert X_train_trans.shape == (50, 16)
    assert X_val_trans.shape == (20, 16)
    # Check that PCA components match
    assert pipeline.pca.n_components_ == 16


# 8. Classifier Training
def test_classifier_training(project_root: Path):
    trainer = ClassicalTrainer(project_root=project_root)

    # Check linear_svm and random_forest initialization
    svm = trainer._build_classifier(classifier_type="linear_svm", calibrate_svm=False)
    rf = trainer._build_classifier(classifier_type="random_forest", rf_n_estimators=10)

    X = np.random.randn(20, 30)
    y = np.array([0] * 10 + [1] * 10)

    svm.fit(X, y)
    rf.fit(X, y)

    assert hasattr(svm, "predict")
    assert hasattr(rf, "predict")
    assert set(svm.classes_) == {0, 1}
    assert set(rf.classes_) == {0, 1}


# 9. Top-5 Predictions
def test_top5_predictions(project_root: Path):
    evaluator = ClassicalEvaluator(project_root=project_root)

    # Mock classifier with known probabilities for 6 classes
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([2, 5])

    class MockCalibratedModel:
        classes_ = np.array([0, 1, 2, 3, 4, 5])

        def predict_proba(self, X):
            return np.array([
                [0.05, 0.10, 0.60, 0.15, 0.05, 0.05],  # Top-1 is class 2
                [0.05, 0.05, 0.10, 0.10, 0.20, 0.50],  # Top-1 is class 5
            ])

        def predict(self, X):
            return np.array([2, 5])

    mock_clf = MockCalibratedModel()
    metrics = evaluator.evaluate(mock_clf, X, y, k=5)
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["top5_accuracy"] == 1.0


# 10. Class Mapping
def test_class_mapping(project_root: Path):
    evaluator = ClassicalEvaluator(project_root=project_root)

    # Class 0 lookup
    info_0 = evaluator.get_class_info(0)
    assert info_0["class_id"] == 0
    assert info_0["original_class_id"] == 1
    assert "Hummer" in info_0["class_name"]
    assert info_0["make"] == "AM General"

    # Out of range fallback
    info_unk = evaluator.get_class_info(9999)
    assert info_unk["class_id"] == 9999
    assert info_unk["class_name"] == "Class_9999"


# 11. Artifact Saving and Loading
def test_artifact_saving_and_loading(synthetic_image: np.ndarray, project_root: Path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pipe_path = tmp_path / "test_pipeline.joblib"
        model_path = tmp_path / "test_model.joblib"

        pipeline = FeatureFusionPipeline(feature_config="color")
        X = np.random.randn(20, 96).astype(np.float32)
        pipeline.fit(X)
        pipeline.save_pipeline(pipe_path)

        clf = DummyClassifier(strategy="constant", constant=0)
        clf.fit(X, np.zeros(20))

        import joblib
        joblib.dump({"classifier": clf, "pipeline": pipeline}, model_path)

        # Reload via Predictor
        predictor = ClassicalVehiclePredictor(
            model_path=model_path,
            pipeline_path=pipe_path,
            project_root=project_root,
        )
        assert predictor.classifier is not None
        assert predictor.pipeline.is_fitted is True

        pred = predictor.predict(synthetic_image)
        assert pred["class_id"] == 0


# 12. Prediction Output Format
def test_prediction_output_format(synthetic_image: np.ndarray, project_root: Path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_path = tmp_path / "dummy_model.joblib"

        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        clf = DummyClassifier(strategy="prior")
        clf.fit(np.zeros((10, 8484)), y)

        pipeline = FeatureFusionPipeline(feature_config="hog_color_lbp")
        pipeline.fit(np.zeros((10, 8484)))

        import joblib
        joblib.dump({"classifier": clf, "pipeline": pipeline}, model_path)

        predictor = ClassicalVehiclePredictor(model_path=model_path, project_root=project_root)
        out = predictor.predict(synthetic_image, top_k=5)

        # Required fields check
        required_keys = [
            "class_id",
            "original_class_id",
            "class_name",
            "make",
            "model",
            "year",
            "confidence",
            "is_calibrated",
            "top_5",
            "feature_extraction_time_ms",
            "classifier_inference_time_ms",
            "total_latency_ms",
        ]
        for key in required_keys:
            assert key in out, f"Missing required output key: {key}"

        assert isinstance(out["top_5"], list)
        assert len(out["top_5"]) == 5
        assert out["top_5"][0]["rank"] == 1
        assert out["feature_extraction_time_ms"] > 0
        assert out["classifier_inference_time_ms"] >= 0


# 13. Edge Cases & Robustness Probing for ROIExtractor
def test_roi_edge_cases(synthetic_image: np.ndarray):
    extractor = ROIExtractor(target_size=(128, 128), keep_aspect_ratio=True, padding_ratio=0.0)

    # 13.1 Out-of-bounds coordinates
    oob_box = (500, 600, 700, 800)
    crop, meta = extractor.extract_roi(synthetic_image, bbox=oob_box)
    assert meta["fallback_used"] is True
    assert crop.shape == (128, 128, 3)

    # 13.2 Negative coordinates completely outside image
    neg_box = (-100, -100, -50, -50)
    crop, meta = extractor.extract_roi(synthetic_image, bbox=neg_box)
    assert meta["fallback_used"] is True

    # 13.3 Inverted coordinates (x2 <= x1 or y2 <= y1)
    inv_box = (200, 150, 100, 50)
    crop, meta = extractor.extract_roi(synthetic_image, bbox=inv_box)
    assert meta["fallback_used"] is True

    # 13.4 Float coordinates
    float_box = [10.4, 20.6, 80.2, 90.7]
    crop, meta = extractor.extract_roi(synthetic_image, bbox=float_box)
    assert meta["fallback_used"] is False
    assert meta["parsed_bbox"] == (10, 21, 81, 92)

    # 13.5 0-based inclusive row vs 1-based MATLAB row
    row_pixel = {"bbox_x1": 38, "bbox_y1": 115, "bbox_x2": 568, "bbox_y2": 374}
    parsed_pixel = extractor.parse_bbox_coordinates(row_pixel, img_width=600, img_height=400)
    assert parsed_pixel == (38, 115, 569, 375)

    row_matlab = {"bbox_matlab_x1": 39, "bbox_matlab_y1": 116, "bbox_matlab_x2": 569, "bbox_matlab_y2": 375}
    parsed_matlab = extractor.parse_bbox_coordinates(row_matlab, img_width=600, img_height=400)
    assert parsed_matlab == (38, 115, 569, 375)

    # 13.6 Degenerate / Empty image handling
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    crop_empty, meta_empty = extractor.extract_roi(empty_img, bbox=(10, 10, 50, 50))
    assert crop_empty.shape == (128, 128, 3)
    assert meta_empty["fallback_used"] is True


# 14. ORB-Only Feature Fusion & Vocabulary Serialization
def test_orb_feature_fusion_and_pipeline_persistence(synthetic_image: np.ndarray):
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe_file = Path(tmpdir) / "orb_pipe.joblib"

        pipeline = FeatureFusionPipeline(feature_config="orb", use_scaler=False, use_pca=False)
        assert pipeline.orb_extractor is not None

        # Pre-fit vocabulary
        pipeline.orb_extractor.fit_vocabulary([synthetic_image, np.fliplr(synthetic_image)], max_descriptors=500)
        assert pipeline.orb_extractor.kmeans is not None

        # Extract features
        feat_1 = pipeline.extract_raw_descriptors(synthetic_image)
        assert feat_1.shape == (pipeline.orb_extractor.vocab_size,)
        assert np.any(feat_1 > 0)

        # Save and reload pipeline
        pipeline.save_pipeline(pipe_file)
        loaded = FeatureFusionPipeline.load_pipeline(pipe_file)

        assert loaded.orb_extractor.kmeans is not None
        feat_2 = loaded.extract_raw_descriptors(synthetic_image)
        np.testing.assert_array_almost_equal(feat_1, feat_2)


# 15. Evaluator Fallback Path with Non-Contiguous Class IDs
def test_evaluator_fallback_no_index_error(project_root: Path):
    evaluator = ClassicalEvaluator(project_root=project_root)

    # Classifier with only predict() and arbitrary high class IDs
    class PredictOnlyClassifier:
        classes_ = np.array([166, 195])

        def predict(self, X):
            return np.array([166, 195])

    X = np.random.randn(2, 64)
    y = np.array([166, 166])

    # Must evaluate without raising IndexError
    metrics = evaluator.evaluate(PredictOnlyClassifier(), X, y)
    assert metrics["top1_accuracy"] == 0.5

    # Failure analysis must also run without IndexError and include visual observations
    failures = evaluator.analyze_failures(PredictOnlyClassifier(), X, y, image_ids=["img_1", "img_2"])
    assert len(failures) == 1
    assert "visual_observations" in failures[0]
    assert len(failures[0]["visual_observations"]) > 10


# 16. Full Top-5 Prediction Columns in Predictor DataFrame
def test_predict_dataframe_full_top5_columns(synthetic_image: np.ndarray, project_root: Path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_path = tmp_path / "dummy_top5_model.joblib"
        pipe_path = tmp_path / "dummy_top5_pipe.joblib"
        img_path = tmp_path / "test_car.png"

        import cv2
        cv2.imwrite(str(img_path), synthetic_image)

        pipeline = FeatureFusionPipeline(feature_config="color")
        X = np.zeros((10, 96))
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        pipeline.fit(X)
        pipeline.save_pipeline(pipe_path)

        clf = DummyClassifier(strategy="prior")
        clf.fit(X, y)

        import joblib
        joblib.dump({"classifier": clf, "pipeline": pipeline}, model_path)

        predictor = ClassicalVehiclePredictor(model_path=model_path, pipeline_path=pipe_path, project_root=project_root)

        import pandas as pd
        df = pd.DataFrame([{
            "image_id": "test_01",
            "image_path": str(img_path),
            "bbox_x1": 10,
            "bbox_y1": 10,
            "bbox_x2": 100,
            "bbox_y2": 100,
        }])

        preds_df = predictor.predict_dataframe(df)
        assert len(preds_df) == 1
        for rank in range(2, 6):
            assert f"top{rank}_class_id" in preds_df.columns
            assert f"top{rank}_score" in preds_df.columns
