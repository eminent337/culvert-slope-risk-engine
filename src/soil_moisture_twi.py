"""
Hyper-Resolution Soil Moisture & Topographic Wetness Index (TWI) Engine.
Downscales regional satellite antecedent soil moisture (SMAP emulation) to hyper-resolution (30m)
using the Beven-Kirkby Topographic Wetness Index and local terrain curvature.
Directly inspired by Dr. Noemi Vergopolan's SMAP-HydroBlocks physical downscaling framework.
"""

import numpy as np
from typing import Tuple, Dict, Any


def compute_topographic_wetness_index(
    accum_cells: np.ndarray,
    slope_radians: np.ndarray,
    resolution: float,
    epsilon: float = 1e-4
) -> np.ndarray:
    """
    Computes the Beven-Kirkby Topographic Wetness Index (TWI):
    
        TWI = ln( a / tan(beta) )
        
    Where:
        a: Specific contributing area per unit contour width (m^2 / m = m).
           a = (accum_cells * resolution^2) / resolution = accum_cells * resolution
        beta: Topographic slope angle in radians.
        tan(beta): Hydraulic gradient under kinematic wave assumption.
        
    Parameters:
        accum_cells: Number of upstream contributing cells (from D8 flow routing).
        slope_radians: Ground slope in radians.
        resolution: Grid cell size in meters.
        epsilon: Regularization parameter to prevent division by zero in flat valleys.
        
    Returns:
        twi: 2D array of Topographic Wetness Index (dimensionless).
    """
    # Specific catchment area per unit contour width (m)
    specific_catchment_area = accum_cells * resolution
    
    # Hydraulic gradient
    tan_beta = np.tan(np.maximum(slope_radians, epsilon))
    
    # Topographic Wetness Index
    twi = np.log(specific_catchment_area / tan_beta)
    
    # Clean non-finite values if any
    twi = np.nan_to_num(twi, nan=0.0, posinf=25.0, neginf=0.0)
    return np.maximum(twi, 0.0)


def downscale_smap_soil_moisture(
    twi: np.ndarray,
    regional_moisture_fraction: float = 0.65,
    twi_sensitivity: float = 0.35
) -> np.ndarray:
    """
    Downscales coarse satellite soil moisture (e.g. NASA SMAP 36km / 9km radiometer data)
    into a hyper-resolution 30m antecedent soil saturation ratio m(x, y).
    
    Mimics the physical downscaling logic of SMAP-HydroBlocks:
    Valleys and convergent hollows with high TWI concentrate moisture (m -> 1.0),
    while divergent ridges and steep knolls drain rapidly (m -> lower).
    
    Parameters:
        twi: 2D array of Topographic Wetness Index.
        regional_moisture_fraction: Basin-wide regional mean saturation ratio (0.0 to 1.0).
        twi_sensitivity: Modulation amplitude of topography on moisture redistribution.
        
    Returns:
        saturation_raster_m: 2D array of local saturation ratio m(x, y) bounded between 0.05 and 1.0.
    """
    p10 = np.percentile(twi, 10)
    p90 = np.percentile(twi, 90)
    
    # Normalized TWI centered around mean zero
    twi_norm = (twi - np.mean(twi)) / max(p90 - p10, 1e-4)
    twi_norm = np.clip(twi_norm, -2.0, 2.0)
    
    # Topographically modulated moisture
    local_m = regional_moisture_fraction * (1.0 + twi_sensitivity * twi_norm)
    
    # Physical constraints: soil saturation ratio must be in [0.05, 1.0]
    saturation_raster_m = np.clip(local_m, 0.05, 1.0)
    return saturation_raster_m


def extract_catchment_spatial_saturation_stats(
    basin_mask: np.ndarray,
    saturation_raster_m: np.ndarray
) -> Dict[str, float]:
    """
    Computes spatial saturation statistics within a culvert's contributing drainage basin.
    """
    if np.sum(basin_mask) == 0:
        return {
            "mean_saturation_m": 0.50,
            "p90_saturation_m": 0.50,
            "saturated_area_fraction": 0.0
        }
        
    basin_m = saturation_raster_m[basin_mask]
    
    return {
        "mean_saturation_m": round(float(np.mean(basin_m)), 3),
        "p90_saturation_m": round(float(np.percentile(basin_m, 90)), 3),
        "saturated_area_fraction": round(float(np.mean(basin_m >= 0.85)), 3)
    }
