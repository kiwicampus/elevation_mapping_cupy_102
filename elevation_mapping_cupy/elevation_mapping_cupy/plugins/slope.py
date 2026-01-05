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
    Slope plugin optimized for stability and persistence of terrain features (like snow).
    
    Processing Stages:
    1. Spatial Filter (Median): Removes high-frequency noise.
    2. Temporal Smoothing (EMA): Rejects transient (single-frame) noise.
    3. Slope Calculation (Arctan2)
    4. Confidence Weighting (Variance + Time Decay): Penalizes old/low-quality data.
    """

    def __init__(
        self,
        output_units: str = "degrees",
        # Filtering Parameters
        max_variance_threshold: float = 0.11, # Max variance threshold (Guardrail). Data > 0.11 is penalized.
        spatial_filter_size: int = 6,         # Kernel size for the spatial median filter (e.g., 6x6).
        temporal_alpha: float = 0.7,          # EMA factor (Alpha). 1.0 = no smoothing, 0.0 = max smoothing.
        time_decay: float = 15.0,             # Time decay factor for confidence (seconds).
        **kwargs,
    ):
        super().__init__()
        self.output_units = output_units.lower()
        if self.output_units not in ['degrees', 'radians']:
            self.output_units = 'degrees'
        
        # Store configuration parameters
        self.max_variance_threshold = float(max_variance_threshold)
        self.spatial_filter_size = spatial_filter_size
        self.temporal_alpha = float(max(0.0, min(1.0, temporal_alpha)))
        self.time_decay = float(time_decay)
        
        # Storage for temporal smoothing (history of normals)
        self.previous_normals: Optional[cp.ndarray] = None
        self.initialized = False
        
        print(f"[Slope] Initialized with Spatial, Temporal Filtering, and Confidence Weighting.")

    def _apply_spatial_filter(self, normal_x: cp.ndarray, normal_y: cp.ndarray, normal_z: cp.ndarray) -> tuple:
        """
        Applies spatial median filtering to the normal vectors.
        Median filter preserves edges better than mean/Gaussian while removing "salt-and-pepper" noise.
        """
        size = self.spatial_filter_size
        nx = ndimage.median_filter(normal_x, size=size)
        ny = ndimage.median_filter(normal_y, size=size)
        nz = ndimage.median_filter(normal_z, size=size)
        return nx, ny, nz

    def _apply_temporal_smoothing(
        self, 
        normal_x: cp.ndarray, 
        normal_y: cp.ndarray, 
        normal_z: cp.ndarray,
        is_valid: cp.ndarray
    ) -> tuple:
        """
        Applies Exponential Moving Average (EMA) to normal vectors for temporal stability.
        This rejects noise peaks that only appear in a single frame.
        """
        current_normals = cp.stack([normal_x, normal_y, normal_z], axis=0)
        
        # Initialize history on the first call
        if not self.initialized or self.previous_normals is None:
            self.previous_normals = current_normals.copy()
            self.initialized = True
            return normal_x, normal_y, normal_z
        
        alpha = self.temporal_alpha
        
        # EMA Formula: smoothed = alpha * current + (1 - alpha) * previous
        # Only apply smoothing to valid cells
        smoothed_normals = cp.where(
            is_valid[None, :, :] > 0.5,
            alpha * current_normals + (1.0 - alpha) * self.previous_normals,
            current_normals
        )
        
        # Update history for the next iteration
        self.previous_normals = smoothed_normals.copy()
        
        return smoothed_normals[0], smoothed_normals[1], smoothed_normals[2]


    def _compute_confidence(
        self, 
        variance: cp.ndarray,
        is_valid: cp.ndarray, 
        time_layer: Optional[cp.ndarray]
    ) -> cp.ndarray:
        """
        Computes the final confidence weight [0, 1] applied to the slope.
        """
        
        # 1. Confidence from Variance (Penalizes low-quality/old data based on threshold)
        c_variance = 1.0 - cp.clip(variance / float(self.max_variance_threshold), 0.0, 1.0)
        
        # 2. Confidence from Validity
        c_valid = cp.where(is_valid > 0.5, 1.0, 0.0)
        
        # 3. Confidence from Data Age (Exponential decay)
        if time_layer is not None and self.time_decay > 0:
            c_age = cp.exp(-time_layer / float(self.time_decay))
        else:
            c_age = 1.0
        
        # Combined Confidence: All factors must be high for high confidence
        return c_variance * c_valid * c_age

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
        
        # 1. Get Auxiliary Layers (variance, is_valid, time, resolution)
        variance = elevation_map[1].copy()
        is_valid = elevation_map[2].copy()
        time_layer = elevation_map[4].copy() if len(elevation_map) > 4 else None
        elevation = elevation_map[0].copy()
        
        # 2. Calculate normals directly from elevation map (more efficient than using pre-computed normals)
        # This avoids the need for dilation_filter_kernel, as the spatial median filter will smooth anyway
        resolution = kwargs.get('resolution', 0.08)  # Default resolution, should be passed from system
        
        # Calculate gradients using finite differences
        # dz/dx and dz/dy for normal calculation
        # We'll calculate this efficiently using array operations
        h, w = elevation.shape
        normal_x = cp.zeros_like(elevation)
        normal_y = cp.zeros_like(elevation)
        normal_z = cp.ones_like(elevation)
        
        # Calculate gradients only for valid cells
        valid_mask = is_valid > 0.5
        # dz/dx: difference in x direction (columns)
        dzdx = cp.zeros_like(elevation)
        dzdy = cp.zeros_like(elevation)
        
        # Forward differences for interior points
        dzdx[:, 1:-1] = cp.where(
            valid_mask[:, 1:-1] & valid_mask[:, 2:],
            (elevation[:, 2:] - elevation[:, 1:-1]) / resolution,
            0.0
        )
        dzdy[1:-1, :] = cp.where(
            valid_mask[1:-1, :] & valid_mask[2:, :],
            (elevation[2:, :] - elevation[1:-1, :]) / resolution,
            0.0
        )
        
        # Normal vectors: n = (-dz/dy, -dz/dx, 1) normalized
        normal_x = -dzdy
        normal_y = -dzdx
        
        # Normalize
        norm = cp.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x = cp.where(valid_mask, normal_x / norm, 0.0)
        normal_y = cp.where(valid_mask, normal_y / norm, 0.0)
        normal_z = cp.where(valid_mask, normal_z / norm, 1.0)
        
        # 3. Stage: Spatial Filter (Smooths out high-frequency sensor noise)
        normal_x, normal_y, normal_z = self._apply_spatial_filter(normal_x, normal_y, normal_z)
        
        # 4. Stage: Temporal Smoothing (Stabilizes normals against transient noise)
        normal_x, normal_y, normal_z = self._apply_temporal_smoothing(
            normal_x, normal_y, normal_z, is_valid
        )
        
        # 5. Calculate Slope (The core mathematical operation)
        # Formula: slope_angle = arctan2(sqrt(nx^2 + ny^2), nz)
        y_term = cp.sqrt(normal_x**2 + normal_y**2)
        slope_rad = cp.arctan2(y_term, normal_z)
        
        slope = slope_rad * 180.0 / cp.pi if self.output_units == 'degrees' else slope_rad

        # 6. Stage: Apply Confidence Weighting (Penalizes low-quality data)
        confidence = self._compute_confidence(variance, is_valid, time_layer)
        
        # Apply confidence to the final slope output
        slope = slope * confidence
        
        return slope