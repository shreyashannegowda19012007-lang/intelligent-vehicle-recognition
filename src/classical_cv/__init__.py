"""
Classical Computer Vision and Machine Learning Baseline Module for Stanford Cars Recognition.
Provides vehicle ROI extraction, HOG/Color/LBP/ORB feature extractors, Bag-of-Visual-Words,
feature fusion pipelines with disk caching, Linear/RBF SVM and Random Forest classifiers,
Top-1/Top-5 evaluation, and visual diagnostics.
"""

from src.classical_cv.color_features import ColorHistogramExtractor
from src.classical_cv.evaluator import ClassicalEvaluator
from src.classical_cv.feature_fusion import FeatureFusionPipeline
from src.classical_cv.hog_features import HOGFeatureExtractor
from src.classical_cv.orb_features import ORBFeatureExtractor
from src.classical_cv.predictor import ClassicalVehiclePredictor
from src.classical_cv.roi_extractor import ROIExtractor
from src.classical_cv.texture_features import LBPFeatureExtractor
from src.classical_cv.trainer import ClassicalTrainer
from src.classical_cv.visualizer import ClassicalVisualizer

__all__ = [
    "ClassicalEvaluator",
    "ClassicalTrainer",
    "ClassicalVehiclePredictor",
    "ClassicalVisualizer",
    "ColorHistogramExtractor",
    "FeatureFusionPipeline",
    "HOGFeatureExtractor",
    "LBPFeatureExtractor",
    "ORBFeatureExtractor",
    "ROIExtractor",
]
