#
# Copyright (c) 2024, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
from cupyx.scipy import ndimage

from typing import List, Optional

from .plugin_manager import PluginBase


class Slope(PluginBase):
    """
    This plugin calculates robust slope angles from surface normal vectors with noise filtering.
    
    Handles camera pose errors and point cloud noise through multi-stage filtering:
    1. Variance filtering - removes high-uncertainty areas
    2. Elevation outlier detection - detects fake walls from pose errors
    3. Spatial median filtering - smooths noisy normals while preserving edges
    4. Temporal smoothing - uses historical data to reject transient noise
    5. Confidence weighting - combines all factors for reliable slopes
    
    The slope is computed using the arctan2 method:
        y_term = sqrt(normal_x^2 + normal_y^2)
        x_term = normal_z
        slope_angle = arctan2(y_term, x_term)

    Args:
        output_units (str): Output units - 'degrees' or 'radians'. Default: 'degrees'
        enable_robust_filtering (bool): Enable multi-stage filtering pipeline. Default: True
        max_variance_threshold (float): Max variance to accept (meters). Default: 0.1
        enable_outlier_detection (bool): Enable elevation outlier detection. Default: True
        outlier_threshold (float): Elevation jump threshold (meters). Default: 0.15
        outlier_kernel_size (int): Kernel size for outlier detection. Default: 5
        enable_spatial_filter (bool): Enable spatial median filtering. Default: True
        spatial_filter_size (int): Median filter kernel size. Default: 3
        enable_temporal_smoothing (bool): Enable temporal smoothing. Default: True
        temporal_alpha (float): Temporal smoothing factor (0=max smooth, 1=no smooth). Default: 0.7
        enable_confidence_weighting (bool): Enable confidence-based weighting. Default: True
        time_decay (float): Time decay factor for confidence (seconds). Default: 2.0
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        output_units: str = "degrees",
        enable_robust_filtering: bool = True,
        max_variance_threshold: float = 0.1,
        enable_outlier_detection: bool = True,
        outlier_threshold: float = 0.15,
        outlier_kernel_size: int = 5,
        enable_spatial_filter: bool = True,
        spatial_filter_size: int = 3,
        enable_temporal_smoothing: bool = True,
        temporal_alpha: float = 0.7,  # Balanced: responsive but noise-rejecting
        enable_confidence_weighting: bool = True,
        time_decay: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.output_units = output_units.lower()
        
        if self.output_units not in ['degrees', 'radians']:
            print(f"Warning: Invalid output_units '{output_units}'. Using 'degrees'.")
            self.output_units = 'degrees'
        
        # Filtering configuration
        self.enable_robust_filtering = enable_robust_filtering
        self.max_variance_threshold = float(max_variance_threshold)
        self.enable_outlier_detection = enable_outlier_detection
        self.outlier_threshold = float(outlier_threshold)
        self.outlier_kernel_size = outlier_kernel_size
        self.enable_spatial_filter = enable_spatial_filter
        self.spatial_filter_size = spatial_filter_size
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.temporal_alpha = float(max(0.0, min(1.0, temporal_alpha)))
        self.enable_confidence_weighting = enable_confidence_weighting
        self.time_decay = float(time_decay)
        
        # Storage for temporal smoothing (initialized on first call)
        self.previous_normals: Optional[cp.ndarray] = None
        self.initialized = False
        
        print(f"Slope plugin initialized with robust filtering: {enable_robust_filtering}")
        if enable_robust_filtering:
            print(f"  - Variance threshold: {max_variance_threshold}m")
            print(f"  - Outlier detection: {enable_outlier_detection} (threshold: {outlier_threshold}m)")
            print(f"  - Spatial filter: {enable_spatial_filter} (size: {spatial_filter_size}x{spatial_filter_size})")
            print(f"  - Temporal smoothing: {enable_temporal_smoothing} (alpha: {temporal_alpha:.2f})")
            print(f"  - Confidence weighting: {enable_confidence_weighting}")

    def _get_elevation_outliers(self, elevation: cp.ndarray, variance: cp.ndarray, is_valid: cp.ndarray) -> cp.ndarray:
        """
        Detect elevation outliers that likely represent pose errors or noise.
        
        Args:
            elevation: Elevation map
            variance: Variance map
            is_valid: Valid cells mask
            
        Returns:
            Boolean mask where True indicates outlier cells
        """
        # Compute local median elevation (robust to outliers)
        kernel_size = self.outlier_kernel_size
        elevation_median = ndimage.median_filter(elevation, size=kernel_size)
        
        # Compute absolute difference from local median
        elevation_diff = cp.abs(elevation - elevation_median)
        
        # Mark as outlier if:
        # 1. Elevation difference exceeds threshold
        # 2. Cell has high variance (uncertain)
        # 3. Cell is valid (we only check valid cells)
        is_outlier = (
            (elevation_diff > float(self.outlier_threshold)) &
            (variance > float(self.max_variance_threshold) / 2.0) &  # Half threshold for combined check
            (is_valid > 0.5)
        )
        
        return is_outlier

    def _apply_spatial_filter(self, normal_x: cp.ndarray, normal_y: cp.ndarray, normal_z: cp.ndarray) -> tuple:
        """
        Apply spatial median filtering to normal vectors.
        Median filter preserves edges better than mean/Gaussian.
        
        Args:
            normal_x, normal_y, normal_z: Normal vector components
            
        Returns:
            Filtered normal vectors (normal_x, normal_y, normal_z)
        """
        size = self.spatial_filter_size
        
        normal_x_filtered = ndimage.median_filter(normal_x, size=size)
        normal_y_filtered = ndimage.median_filter(normal_y, size=size)
        normal_z_filtered = ndimage.median_filter(normal_z, size=size)
        
        return normal_x_filtered, normal_y_filtered, normal_z_filtered

    def _apply_temporal_smoothing(
        self, 
        normal_x: cp.ndarray, 
        normal_y: cp.ndarray, 
        normal_z: cp.ndarray,
        is_valid: cp.ndarray
    ) -> tuple:
        """
        Apply exponential moving average to normal vectors for temporal stability.
        
        Args:
            normal_x, normal_y, normal_z: Current normal vectors
            is_valid: Valid cells mask
            
        Returns:
            Temporally smoothed normal vectors
        """
        # Stack current normals
        current_normals = cp.stack([normal_x, normal_y, normal_z], axis=0)
        
        if not self.initialized or self.previous_normals is None:
            # First call - no smoothing, just store
            self.previous_normals = current_normals.copy()
            self.initialized = True
            return normal_x, normal_y, normal_z
        
        # Exponential moving average: smoothed = alpha * current + (1-alpha) * previous
        # Higher alpha = more responsive to changes (detect new obstacles faster)
        # Lower alpha = more smoothing (reject noise better)
        alpha = self.temporal_alpha
        
        # Only smooth valid cells; invalid cells take current values
        smoothed_normals = cp.where(
            is_valid[None, :, :] > 0.5,  # Broadcast is_valid to shape (3, H, W)
            alpha * current_normals + (1.0 - alpha) * self.previous_normals,
            current_normals
        )
        
        # Update history
        self.previous_normals = smoothed_normals.copy()
        
        return smoothed_normals[0], smoothed_normals[1], smoothed_normals[2]

    def _compute_confidence(
        self,
        variance: cp.ndarray,
        is_valid: cp.ndarray,
        is_outlier: cp.ndarray,
        time_layer: Optional[cp.ndarray] = None
    ) -> cp.ndarray:
        """
        Compute confidence weights for slope values.
        
        Args:
            variance: Variance layer
            is_valid: Valid cells mask  
            is_outlier: Outlier cells mask
            time_layer: Time since last update (optional)
            
        Returns:
            Confidence weights [0, 1]
        """
        # Confidence from variance (low variance = high confidence)
        c_variance = 1.0 - cp.clip(variance / float(self.max_variance_threshold), 0.0, 1.0)
        
        # Confidence from validity
        c_valid = cp.where(is_valid > 0.5, 1.0, 0.0)
        
        # Confidence from outlier detection (outliers get zero confidence)
        c_outlier = cp.where(is_outlier, 0.0, 1.0)
        
        # Confidence from data age (fresher data = higher confidence)
        if time_layer is not None and self.time_decay > 0:
            c_age = cp.exp(-time_layer / float(self.time_decay))
        else:
            c_age = 1.0
        
        # Combined confidence
        confidence = c_variance * c_valid * c_outlier * c_age
        
        return confidence

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        *args,
        **kwargs,
    ) -> cp.ndarray:
        """
        Calculates robust slope angles from normal vectors with multi-stage filtering.

        Args:
            elevation_map (cupy._core.core.ndarray): Elevation map with layers.
            layer_names (List[str]): Layer names in elevation map.
            plugin_layers (cupy._core.core.ndarray): Plugin layers.
            plugin_layer_names (List[str]): Plugin layer names.
            semantic_map (cupy._core.core.ndarray): Semantic map layers.
            semantic_layer_names (List[str]): Semantic layer names.
            *args: Additional arguments.

        Returns:
            cupy._core.core.ndarray: Slope map in degrees [0, 90] or radians [0, π/2].
        """
        # Get normal vectors from kwargs (passed from elevation_mapping.py)
        normal_map = kwargs.get('normal_map', None)
        
        if normal_map is not None:
            # Use existing normals from the system
            if not hasattr(self, '_logged_method'):
                print("[Slope] Using system-provided normal map (Efficient)")
                self._logged_method = True
                
            normal_x = normal_map[0].copy()
            normal_y = normal_map[1].copy()
            normal_z = normal_map[2].copy()
        else:
            # Fallback: Compute normals from elevation if not provided
            if not hasattr(self, '_logged_method'):
                print("[Slope] Warning: System normals not found. Computing from elevation gradient (Fallback)")
                self._logged_method = True
                
            elevation = elevation_map[0].copy()
            resolution = 0.07  # Should match config
            grad_y, grad_x = cp.gradient(elevation, resolution)
            normal_x = -grad_x
            normal_y = -grad_y
            normal_z = cp.ones_like(elevation)
            
            # Normalize
            norm = cp.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
            normal_x = normal_x / (norm + 1e-8)
            normal_y = normal_y / (norm + 1e-8)
            normal_z = normal_z / (norm + 1e-8)
            
        # Get additional layers for robust filtering
        if self.enable_robust_filtering:
            # Get elevation-related layers
            elevation = elevation_map[0].copy() # elevation layer, needed for outlier detection
            variance = elevation_map[1].copy()   # variance
            is_valid = elevation_map[2].copy()   # is_valid
            time_layer = elevation_map[4].copy() if len(elevation_map) > 4 else None  # time
            
            # Stage 1: Variance-based filtering
            high_variance_mask = variance > float(self.max_variance_threshold)
            
            # Stage 2: Elevation outlier detection
            if self.enable_outlier_detection:
                is_outlier = self._get_elevation_outliers(elevation, variance, is_valid)
            else:
                is_outlier = cp.zeros_like(is_valid, dtype=bool)
            
            # Stage 3: Spatial median filtering
            if self.enable_spatial_filter:
                normal_x, normal_y, normal_z = self._apply_spatial_filter(
                    normal_x, normal_y, normal_z
                )
            
            # Stage 4: Temporal smoothing
            if self.enable_temporal_smoothing:
                normal_x, normal_y, normal_z = self._apply_temporal_smoothing(
                    normal_x, normal_y, normal_z, is_valid
                )
        else:
            # No filtering - use raw data
            variance = None
            is_valid = cp.ones_like(normal_x)
            is_outlier = cp.zeros_like(normal_x, dtype=bool)
            time_layer = None
        
        # Calculate slope using arctan2 (robust method)
        y_term = cp.sqrt(normal_x**2 + normal_y**2)
        x_term = normal_z
        slope_rad = cp.arctan2(y_term, x_term)
        
        # Convert to degrees if requested
        if self.output_units == 'degrees':
            slope = slope_rad * 180.0 / cp.pi
        else:
            slope = slope_rad

        # DEBUG: Print slope stats
        # print(f"  Slope (raw): min={float(cp.min(slope)):.3f}, max={float(cp.max(slope)):.3f}, mean={float(cp.mean(slope)):.3f}")
        
        # Stage 5: Confidence weighting
        if self.enable_robust_filtering and self.enable_confidence_weighting:
            confidence = self._compute_confidence(variance, is_valid, is_outlier, time_layer)
            # Apply confidence: low-confidence areas get reduced slope values
            # This makes unreliable slopes less prominent
            slope = slope * confidence
            # print(f"  Slope (filtered): min={float(cp.min(slope)):.3f}, max={float(cp.max(slope)):.3f}, mean={float(cp.mean(slope)):.3f}")
        
        return slope
