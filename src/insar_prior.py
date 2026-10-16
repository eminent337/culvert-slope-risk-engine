"""
Sentinel-1 Multi-Temporal InSAR Kinematic Observational Prior Module.
Integrates satellite radar interferometry line-of-sight (LOS) ground deformation velocities (mm/yr)
to update static geotechnical slope stability models into dynamic observationally-constrained risk indices.
Draws directly on Peter Aryee's Chirano Gold Mine Sentinel-1 InSAR capstone thesis (90.5%).
"""

import numpy as np
from typing import Dict, Any, Tuple


def generate_synthetic_sentinel1_insar_field(
    slope_deg: np.ndarray,
    saturation_raster_m: np.ndarray,
    seed: int = 42
) -> np.ndarray:
    """
    Simulates a realistic Sentinel-1 InSAR surface deformation velocity field (v_LOS in mm/year).
    
    Physical coupling:
    Steep slopes (>30 deg) with high antecedent moisture saturation (m > 0.70) experience
    elevated gravitational shear creep, producing measurable negative line-of-sight velocities
    (downhill movement away from the satellite sensor).
    
    Parameters:
        slope_deg: 2D array of slope gradients in degrees.
        saturation_raster_m: 2D array of local saturation ratio.
        seed: Random seed for geodetic atmospheric phase screen (APS) noise.
        
    Returns:
        v_los_mm_yr: 2D array of Sentinel-1 LOS velocity in mm/year (negative = subsidence/downslope creep).
    """
    np.random.seed(seed)
    rows, cols = slope_deg.shape
    
    # 1. Geodetic noise floor (Atmospheric Phase Screen & sensor noise ~ N(0, 1.8 mm/yr))
    noise = np.random.normal(0, 1.5, (rows, cols))
    
    # 2. Kinematic downslope creep acceleration
    # Threshold behavior: creep initiates when slope > 28 deg and saturation m > 0.65
    creep_potential = np.maximum(0.0, (slope_deg - 25.0) / 15.0) * np.maximum(0.0, (saturation_raster_m - 0.55) / 0.45)
    
    # Non-linear velocity response (up to -28 mm/yr creep in highly saturated steep gullies)
    creep_velocity = -28.0 * (creep_potential ** 1.8)
    
    v_los_mm_yr = creep_velocity + noise
    return v_los_mm_yr.astype(np.float32)


def update_factor_of_safety_with_insar(
    fs_prior: float,
    v_los_mm_yr: float,
    noise_threshold_mm_yr: float = 3.5,
    velocity_scale_mm_yr: float = 18.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Updates the static geotechnical Factor of Safety (FS) using dynamic InSAR deformation observations.
    
    Formulation (Kinematic Degradation Function):
        v_excess = max(0, |v_los| - noise_threshold)
        FS_insar = FS_prior * exp( - (v_excess / velocity_scale) )
        
    If |v_los| <= noise_threshold:
        Slope is stationary; FS_insar = FS_prior.
    If |v_los| >> noise_threshold:
        Active physical deformation confirms strain softening / creeping slip plane,
        exponentially reducing the operational Factor of Safety.
        
    Parameters:
        fs_prior: Static Terzaghi-Coulomb Factor of Safety.
        v_los_mm_yr: Measured Sentinel-1 LOS displacement rate in mm/year.
        noise_threshold_mm_yr: Geodetic InSAR noise floor below which motion is non-significant.
        velocity_scale_mm_yr: Characteristic deformation velocity scaling parameter.
        
    Returns:
        fs_updated: InSAR-updated Factor of Safety.
        insar_metadata: Diagnostic dictionary detailing deformation kinematics.
    """
    abs_v = abs(v_los_mm_yr)
    v_excess = max(0.0, abs_v - noise_threshold_mm_yr)
    
    # Kinematic degradation multiplier
    degradation_factor = np.exp(- (v_excess / velocity_scale_mm_yr))
    fs_updated = round(float(fs_prior * degradation_factor), 2)
    
    # InSAR Activity Classification
    if abs_v < 4.0:
        kinematic_status = "Stationary (Stable Crust)"
        activity_level = "Baseline"
    elif abs_v < 10.0:
        kinematic_status = "Slow Downslope Creep"
        activity_level = "Moderate"
    elif abs_v < 20.0:
        kinematic_status = "Accelerated Deformation"
        activity_level = "Elevated"
    else:
        kinematic_status = "Active Slope Displacement / Imminent Failure"
        activity_level = "Severe"
        
    metadata = {
        "v_los_mm_yr": round(v_los_mm_yr, 2),
        "kinematic_status": kinematic_status,
        "activity_level": activity_level,
        "kinematic_degradation_pct": round(float((1.0 - degradation_factor) * 100.0), 1),
        "fs_prior": fs_prior,
        "fs_insar_updated": fs_updated
    }
    
    return fs_updated, metadata
