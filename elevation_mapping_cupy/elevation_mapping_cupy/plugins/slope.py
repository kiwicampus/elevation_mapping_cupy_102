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
        spatial_filter_size: int = 6,         # Kernel size for the spatial filter (e.g., 6x6). Set <=1 to disable.
        spatial_filter_type: str = "median",  # "median" (robust but expensive), "uniform" (fast), "none"
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
        self.spatial_filter_type = str(spatial_filter_type).lower()
        self.temporal_alpha = float(max(0.0, min(1.0, temporal_alpha)))
        self.time_decay = float(time_decay)
        
        # Storage for temporal smoothing (history of normals)
        self.previous_normals: Optional[cp.ndarray] = None
        self.initialized = False

        print("[Slope] Initialized with Spatial, Temporal Filtering, and Confidence Weighting.")

    def _apply_spatial_filter(self, normal_x: cp.ndarray, normal_y: cp.ndarray, normal_z: cp.ndarray) -> tuple:
        """
        Applies spatial filtering to the normal vectors.
        - median: robust to spikes but expensive
        - uniform: fast smoothing
        - none / size<=1: disabled
        """
        size = self.spatial_filter_size
        if size is None or int(size) <= 1 or self.spatial_filter_type in ("none", "off", "false", "0"):
            return normal_x, normal_y, normal_z

        if self.spatial_filter_type == "uniform":
            nx = ndimage.uniform_filter(normal_x, size=size)
            ny = ndimage.uniform_filter(normal_y, size=size)
            nz = ndimage.uniform_filter(normal_z, size=size)
        else:
            # default: median
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
            self.previous_normals = current_normals
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
        self.previous_normals = smoothed_normals
        
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
        
        # 1. Get Normal Vectors from System (Assumed to be provided by the elevation mapping core)
        normal_map = kwargs.get('normal_map', None)
        if normal_map is None:
            raise ValueError("[Slope] normal_map was not provided. Ensure ElevationMap passes normal_map into PluginManager.update_with_name().")
        # Avoid unnecessary copies; filters will allocate outputs as needed.
        normal_x, normal_y, normal_z = normal_map[0], normal_map[1], normal_map[2]
            
        # 2. Get Auxiliary Layers (variance, is_valid, time)
        # These layers are crucial for robust filtering and confidence weighting.
        variance = elevation_map[1]
        is_valid = elevation_map[2]
        time_layer = elevation_map[4] if len(elevation_map) > 4 else None
        
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