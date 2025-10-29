#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.features.hog_extractor import MultiScaleHOGExtractor, AdaptiveHOGExtractor
from src.features.lbp_extractor import ComprehensiveLBPExtractor
from src.features.gabor_extractor import GaborFeatureExtractor, AdaptiveGaborExtractor

logger = logging.getLogger(__name__)


class CombinedConfig:
    """Unified config (no dataclass / no type hints)."""

    def __init__(self, **kwargs):
        # toggles
        self.use_hog = kwargs.get("use_hog", True)
        self.use_lbp = kwargs.get("use_lbp", True)
        self.use_gabor = kwargs.get("use_gabor", True)
        self.use_sfs = kwargs.get("use_sfs", True)

        # parallel
        self.max_workers = int(kwargs.get("max_workers", 4))

        # HOG
        self.hog_mode = str(kwargs.get("hog_mode", "multiscale"))

        # LBP
        self.lbp_radius_list = kwargs.get("lbp_radius_list", None)

        # Gabor
        self.gabor_mode = kwargs.get("gabor_mode", "optimized")  # may be str/tuple/list
        self.gabor_n_frequencies = int(kwargs.get("gabor_n_frequencies", kwargs.get("gabor_freq", 6)))
        self.gabor_n_orientations = int(kwargs.get("gabor_n_orientations", kwargs.get("gabor_ori", 8)))
        self.gabor_patch_size = int(kwargs.get("gabor_patch_size", kwargs.get("gabor_patch", 32)))
        self.gabor_compute_phase = bool(kwargs.get("gabor_compute_phase", kwargs.get("gabor_phase", False)))

        # SfS
        self.sfs_config = kwargs.get("sfs_config", None)


class ComprehensiveFeatureExtractor:
    """Top-level feature combiner (HOG/LBP/Gabor)."""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else CombinedConfig()

        # HOG
        self.hog_multiscale = None
        self.hog_adaptive = None
        if self.cfg.use_hog:
            if str(self.cfg.hog_mode).lower() == "multiscale":
                self.hog_multiscale = MultiScaleHOGExtractor()
            else:
                self.hog_adaptive = AdaptiveHOGExtractor()

        # LBP
        if self.cfg.use_lbp:
            radii = self.cfg.lbp_radius_list or [2]
            self.lbp_comprehensive = ComprehensiveLBPExtractor(
                radius_list=radii,
                points_list=[16] * len(radii)
            )
        else:
            self.lbp_comprehensive = None

        # Gabor mode normalization
        gm = self.cfg.gabor_mode
        if isinstance(gm, (tuple, list)):  # just in case
            gm = gm[0] if len(gm) > 0 else "optimized"
        gm = str(gm).lower()

        # Gabor
        self.gabor = None
        self.gabor_adaptive = None
        if self.cfg.use_gabor:
            if gm == "adaptive":
                self.gabor_adaptive = AdaptiveGaborExtractor()
                logger.info(f"[INIT] Gabor mode=adaptive")
            else:
                fmin = 0.02 if gm == "optimized" else 0.01
                fmax = 0.35 if gm == "optimized" else 0.30
                self.gabor = GaborFeatureExtractor(
                    n_frequencies=self.cfg.gabor_n_frequencies,
                    n_orientations=self.cfg.gabor_n_orientations,
                    patch_size=self.cfg.gabor_patch_size,
                    compute_phase=self.cfg.gabor_compute_phase,
                    frequency_min=fmin,
                    frequency_max=fmax,
                    use_parallel=True
                )
                logger.info(f"[INIT] Gabor mode={gm} (freq_range={fmin}-{fmax})")

        logger.info("[INIT] ComprehensiveFeatureExtractor ready.")

    # ----------------------------
    # Per-feature wrappers
    # ----------------------------
    def _feat_hog(self, img):
        if not self.cfg.use_hog:
            return np.array([], dtype=np.float32)
        try:
            if self.hog_multiscale is not None:
                return self.hog_multiscale.extract_combined_features(img).astype(np.float32)
            return self.hog_adaptive.extract_adaptive_features(img).astype(np.float32)
        except Exception as e:
            logger.warning(f"[WARN] HOG extraction failed: {e}")
            return np.array([], dtype=np.float32)

    def _feat_lbp(self, img):
        if not self.cfg.use_lbp:
            return np.array([], dtype=np.float32)
        try:
            return self.lbp_comprehensive.extract_comprehensive_lbp_features(img).astype(np.float32)
        except Exception as e:
            logger.warning(f"[WARN] LBP extraction failed: {e}")
            return np.array([], dtype=np.float32)

    def _feat_gabor(self, img):
        if not self.cfg.use_gabor:
            return np.array([], dtype=np.float32)
        try:
            if self.gabor_adaptive is not None:
                return self.gabor_adaptive.extract_adaptive_features(img).astype(np.float32)
            return self.gabor.extract_comprehensive_features(img).astype(np.float32)
        except Exception as e:
            logger.warning(f"[WARN] Gabor extraction failed: {e}")
            return np.array([], dtype=np.float32)

    def _feat_sfs(self, img):
        if not self.cfg.use_sfs:
            return np.array([], dtype=np.float32)
        try:
            return self.sfs.extract_comprehensive_sfs_features(img).astype(np.float32)
        except Exception as e:
            logger.warning(f"[WARN] SfS extraction failed: {e}")
            return np.array([], dtype=np.float32)

    # ----------------------------
    # Public API
    # ----------------------------
    def extract_single(self, image):
        out = {}
        if self.cfg.use_hog:   out["hog"]   = self._feat_hog(image)
        if self.cfg.use_lbp:   out["lbp"]   = self._feat_lbp(image)
        if self.cfg.use_gabor: out["gabor"] = self._feat_gabor(image)
        if self.cfg.use_sfs:   out["sfs"]   = self._feat_sfs(image)
        return out

    def extract_batch(self, images):
        N = len(images)
        active = [k for k in ["hog", "lbp", "gabor", "sfs"] if getattr(self.cfg, f"use_{k}", False)]
        if N == 0:
            return {k: np.zeros((0, 0), dtype=np.float32) for k in active}

        def worker(idx):
            img = images[idx]
            feats = self.extract_single(img)
            return idx, feats

        per_image = [None] * N
        max_workers = max(1, int(self.cfg.max_workers))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker, i) for i in range(N)]
            for fut in as_completed(futures):
                i, feats = fut.result()
                per_image[i] = feats

        blocks = {}
        for i in range(N):
            feats = per_image[i] or {}
            for name in active:
                v = feats.get(name, np.array([], dtype=np.float32))
                blocks.setdefault(name, []).append(v)

        out = {}
        for name, vecs in blocks.items():
            max_d = max((len(v) for v in vecs), default=0)
            mat = np.zeros((N, max_d), dtype=np.float32)
            for i, v in enumerate(vecs):
                if v is None or len(v) == 0:
                    continue
                d = min(len(v), max_d)
                mat[i, :d] = v[:d]
            out[name] = mat

        return out
