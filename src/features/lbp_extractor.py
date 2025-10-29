# lbp_extractor.py
"""
LBP-based feature extractors (uniform, rotation-invariant, and a comprehensive wrapper).

This module provides different LBP variants for texture analysis.
"""

import logging
import numpy as np
import cv2
from skimage.feature import local_binary_pattern

logger = logging.getLogger(__name__)

class UniformLBP:
    """
    Extracts a histogram of uniform LBP codes. This is effective for general texture classification.
    """
    def __init__(self, radius_list: list = [2], points_list: list = [16], blur_ksize: int = 3):
        self.radius_list = radius_list
        self.points_list = points_list
        self.blur_ksize = blur_ksize

    def extract(self, image: np.ndarray) -> np.ndarray:
        feats = []
        
        # Apply a gentle blur to reduce noise sensitivity before LBP calculation
        blurred_image = cv2.GaussianBlur(image, (self.blur_ksize, self.blur_ksize), 0)

        for R, P in zip(self.radius_list, self.points_list):
            lbp = local_binary_pattern(blurred_image, P, R, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist /= hist.sum()
            feats.append(hist)
        return np.concatenate(feats, axis=0) if feats else np.array([], dtype=np.float32)

# ---------------------- Rotation Invariant LBP ----------------------

class RotationInvariantLBP:
    def __init__(self, radius_list=[2], points_list=[16]):
        self.radius_list = radius_list
        self.points_list = points_list

    def extract(self, image: np.ndarray) -> np.ndarray:
        feats = []
        for R, P in zip(self.radius_list, self.points_list):
            lbp = local_binary_pattern(image, P, R, method="ror")
            hist, _ = np.histogram(lbp.ravel(),
                                   bins=np.arange(0, P + 3),
                                   range=(0, P + 2))
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist /= hist.sum()
            feats.append(hist)
        return np.concatenate(feats, axis=0) if feats else np.array([], dtype=np.float32)


# ---------------------- Comprehensive Extractor ----------------------

class ComprehensiveLBPExtractor:
    """
    A wrapper for LBP feature extraction, designed to be the main entry point.
    This class is designed to be the main entry point for LBP feature extraction.
    """

    def __init__(self,
                 radius_list: list = [2], points_list: list = [16]):
        self.radius_list = radius_list
        self.points_list = points_list

        # Initialize the uniform LBP extractor with multi-radius settings
        self.uniform_extractor = UniformLBP(radius_list, points_list)

        logger.info(f"[INIT] ComprehensiveLBPExtractor - Radii={self.radius_list}, Points={self.points_list}")

    def extract_comprehensive_lbp_features(self,
                                   image: np.ndarray) -> np.ndarray:
        features = []

        # Extract uniform LBP features (good for general texture)
        uniform_feats = self.uniform_extractor.extract(image)
        if uniform_feats.size > 0:
            features.append(uniform_feats)

        if features:
            feats = np.concatenate(features, axis=0).astype(np.float32)
            logger.info(f"[INFO] LBP feature dim: {feats.size}")  # log before returning
            return feats

        return np.array([], dtype=np.float32)
