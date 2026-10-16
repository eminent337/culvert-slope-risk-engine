"""
USDA Forest Service Debris & Sediment Plugging Hazard Engine.
Implements woody debris jamming probability and sediment delivery constriction models
derived from USDA Forest Service GRAIP (Geomorphic Road Analysis and Inventory Package)
and empirical culvert plugging research (Furniss et al., 1998; Flanagan et al., 2004).
"""

import numpy as np
from typing import Dict, Any, Tuple


def estimate_bankfull_channel_width(catchment_area_km2: float) -> float:
    """
    Estimates natural bankfull channel width (W_bf in meters) from contributing drainage area
    using standard regional hydraulic geometry power-law scaling (Dunne & Leopold):
    
        W_bf = 1.9 * (A_catchment)^0.38
    """
    w_bf = 1.9 * (max(catchment_area_km2, 0.01) ** 0.38)
    return float(max(w_bf, 0.8))


def calculate_debris_jamming_probability(
    pipe_diameter_m: float,
    catchment_area_km2: float,
    stream_slope_deg: float,
    forest_cover_fraction: float = 0.85
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates the probability of woody debris and coarse sediment wedging across the culvert inlet.
    
    Parameters:
        pipe_diameter_m: Inside diameter of the culvert barrel in meters.
        catchment_area_km2: Drainage basin area in square kilometers.
        stream_slope_deg: Channel gradient in degrees.
        forest_cover_fraction: Riparian forest density supplying large woody debris (LWD).
        
    Returns:
        p_plugging: Probability of debris plugging (0.0 to 1.0).
        diagnostics: Dictionary containing constriction ratio and stream power.
    """
    w_bf = estimate_bankfull_channel_width(catchment_area_km2)
    
    # 1. Inlet Constriction Ratio (chi = D_pipe / W_bankfull)
    # If culvert is narrower than the natural stream (chi < 1.0), logs easily span and jam the inlet.
    constriction_ratio = pipe_diameter_m / w_bf
    
    # 2. Stream Power Proxy (SPI = ln(Area * tan(beta)))
    slope_rad = np.radians(max(stream_slope_deg, 0.5))
    tan_slope = np.tan(slope_rad)
    stream_power_index = np.log(max(catchment_area_km2 * 1000.0 * tan_slope, 0.1))
    
    # Logistic plugging probability function
    # Low constriction ratio (pipe narrower than stream) + high stream power = severe jamming risk
    logit = 1.8 * (1.0 - constriction_ratio) + 0.35 * (stream_power_index - 3.0) + 1.2 * (forest_cover_fraction - 0.5)
    p_plugging = 1.0 / (1.0 + np.exp(-logit))
    p_plugging = float(np.clip(p_plugging, 0.05, 0.95))
    
    # Effective conveyance reduction
    # When debris lodges in the barrel inlet, effective hydraulic capacity drops sharply
    capacity_retention_factor = float(1.0 - (0.65 * p_plugging))
    
    if p_plugging >= 0.70:
        plugging_tier = "Severe Debris Jamming Risk"
    elif p_plugging >= 0.40:
        plugging_tier = "Moderate Debris Risk"
    else:
        plugging_tier = "Low Debris Obstruction Risk"
        
    diagnostics = {
        "bankfull_width_m": round(w_bf, 2),
        "constriction_ratio": round(constriction_ratio, 2),
        "stream_power_index": round(float(stream_power_index), 2),
        "plugging_probability": round(p_plugging, 3),
        "capacity_retention_factor": round(capacity_retention_factor, 2),
        "plugging_tier": plugging_tier
    }
    
    return p_plugging, diagnostics
