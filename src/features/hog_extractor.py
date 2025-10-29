import numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from skimage.feature import hog
from skimage import exposure
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HOGConfig:
    orientations: int = 9
    pixels_per_cell: Tuple[int, int] = (8, 8)
    cells_per_block: Tuple[int, int] = (2, 2)
    block_norm: str = 'L2-Hys'
    visualize: bool = False
    transform_sqrt: bool = False
    feature_vector: bool = True


class MultiScaleHOGExtractor:
    def __init__(self, scales: Optional[List[HOGConfig]] = None):
        if scales is None:
            self.scales = [
                # Medium scale for general shapes
                HOGConfig(
                    orientations=9, 
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys'
                ),
                # Coarse scale for larger structures
                HOGConfig(
                    orientations=9,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys'
                ),
                # Even coarser scale for very general structure
                HOGConfig(
                    orientations=9,
                    pixels_per_cell=(24, 24),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys'
                )
            ]
        else:
            self.scales = scales
            
        logger.info(f"[INIT] MultiScaleHOGExtractor - {len(self.scales)} scales")
    
    def extract_single_scale(self, image: np.ndarray, config: HOGConfig) -> np.ndarray:
        try:
            if len(image.shape) == 3:
                if CV2_AVAILABLE:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
            
            image = exposure.equalize_adapthist(image)
            
            features = hog(
                image,
                orientations=config.orientations,
                pixels_per_cell=config.pixels_per_cell,
                cells_per_block=config.cells_per_block,
                block_norm=config.block_norm,
                visualize=config.visualize,
                transform_sqrt=config.transform_sqrt,
                feature_vector=config.feature_vector
            )
            
            if config.visualize:
                features = features[0]
            
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"HOG feature extraction failed: {e}")
            return np.array([], dtype=np.float32)
    
    def extract_multiscale_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        features = {}
        
        for i, config in enumerate(self.scales):
            scale_name = f"scale_{i}"
            
            try:
                scale_features = self.extract_single_scale(image, config)
                features[scale_name] = scale_features
                logger.debug(f"{scale_name}: Extracted {len(scale_features)}-dimensional features")
                
            except Exception as e:
                logger.warning(f"{scale_name} HOG extraction failed: {e}")
                features[scale_name] = np.array([], dtype=np.float32)
        
        return features
    
    def extract_combined_features(self, image: np.ndarray, scales_override: Optional[List[HOGConfig]] = None) -> np.ndarray:
        """Combines and returns HOG features from all scales."""
        original_scales = self.scales
        if scales_override:
            self.scales = scales_override
        
        multiscale_features = self.extract_multiscale_features(image)
        
        valid_features = [features for features in multiscale_features.values() 
                         if len(features) > 0]
        
        if not valid_features:
            logger.warning("No valid HOG features were extracted.")
            combined = np.array([], dtype=np.float32)
        else:
            combined = np.concatenate(valid_features)
        
        # Restore original scales if they were overridden
        if scales_override:
            self.scales = original_scales
            
        logger.debug(f"Combined HOG feature dimension: {len(combined)}")
        return combined
    
    def extract_with_visualization(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        features = []
        visualizations = []
        
        for i, config in enumerate(self.scales):
            config_viz = HOGConfig(
                orientations=config.orientations,
                pixels_per_cell=config.pixels_per_cell,
                cells_per_block=config.cells_per_block,
                block_norm=config.block_norm,
                visualize=True
            )
            
            try:
                processed_image = image.copy()
                if len(processed_image.shape) == 3:
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                
                processed_image = exposure.equalize_adapthist(processed_image)
                
                hog_features, hog_image = hog(
                    processed_image,
                    orientations=config_viz.orientations,
                    pixels_per_cell=config_viz.pixels_per_cell,
                    cells_per_block=config_viz.cells_per_block,
                    block_norm=config_viz.block_norm,
                    visualize=True,
                    feature_vector=True
                )
                
                features.append(hog_features)
                visualizations.append(hog_image)
                
            except Exception as e:
                logger.warning(f"Scale {i} HOG visualization failed: {e}")
        
        if features:
            combined_features = np.concatenate(features)
        else:
            combined_features = np.array([], dtype=np.float32)
        
        return combined_features, visualizations


class AdaptiveHOGExtractor:
    def __init__(self):
        self.base_extractor = MultiScaleHOGExtractor()
        
    def analyze_image_characteristics(self, image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        characteristics = {
            'contrast': np.std(gray),
            'brightness': np.mean(gray),
            'edge_density': self._calculate_edge_density(gray),
            'texture_complexity': self._calculate_texture_complexity(gray)
        }
        
        return characteristics
    
    def _calculate_edge_density(self, image: np.ndarray) -> float:
        edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
        return np.sum(edges > 0) / edges.size
    
    def _calculate_texture_complexity(self, image: np.ndarray) -> float:
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return laplacian.var()
    
    def select_optimal_config(self, characteristics: Dict[str, float]) -> List[HOGConfig]:
        configs = []
        
        if characteristics['contrast'] < 0.1:
            orientations = 12
        else:
            orientations = 9
        
        # Adjust cell size based on edge density, but avoid overly fine cells to improve generalization.
        if characteristics['edge_density'] > 0.05:
            # For high edge density, use medium and coarse cells.
            cell_sizes = [(8, 8), (16, 16)]
        else:
            # For low edge density, focus on coarser cells.
            cell_sizes = [(12, 12), (24, 24)]
        
        for cell_size in cell_sizes:
            config = HOGConfig(
                orientations=orientations,
                pixels_per_cell=cell_size,
                cells_per_block=(2, 2),
                block_norm='L2' # L2-Hys can be too aggressive
            )
            configs.append(config)
        
        return configs
    
    def extract_adaptive_features(self, image: np.ndarray) -> np.ndarray:
        characteristics = self.analyze_image_characteristics(image)
        optimal_configs = self.select_optimal_config(characteristics)
        # Reuse the base extractor by passing the optimal configs dynamically
        features = self.base_extractor.extract_combined_features(image, scales_override=optimal_configs)
        logger.info(f"Adaptive HOG features extracted: {len(features)} dimensions")
        return features

class HOGFeatureSelector:
    def __init__(self, selection_method: str = 'variance'):
        self.selection_method = selection_method
        self.selected_indices = None
        
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None, top_k: int = 100):
        if self.selection_method == 'variance':
            variances = np.var(features, axis=0)
            self.selected_indices = np.argsort(variances)[-top_k:]
            
        elif self.selection_method == 'mutual_info' and labels is not None:
            from sklearn.feature_selection import mutual_info_classif
            scores = mutual_info_classif(features, labels)
            self.selected_indices = np.argsort(scores)[-top_k:]
            
        elif self.selection_method == 'correlation' and labels is not None:
            correlations = []
            for i in range(features.shape[1]):
                corr = np.corrcoef(features[:, i], labels)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            
            self.selected_indices = np.argsort(correlations)[-top_k:]
            
        else:
            self.selected_indices = np.arange(min(top_k, features.shape[1]))
        
        logger.info(f"HOG feature selection complete: {len(self.selected_indices)} features selected.")
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.selected_indices is None:
            raise ValueError("The fit() method must be called first.")
        
        return features[:, self.selected_indices]
    
    def fit_transform(self, features: np.ndarray, labels: Optional[np.ndarray] = None, 
                     top_k: int = 100) -> np.ndarray:
        self.fit(features, labels, top_k)
        return self.transform(features)