"""
Data preprocessing, annotation parsing, and PyTorch dataset modules.
"""

from .dataset import StanfordCarsDataset
from .parser import parse_annotations, parse_cars_meta

__all__ = ["StanfordCarsDataset", "parse_annotations", "parse_cars_meta"]
