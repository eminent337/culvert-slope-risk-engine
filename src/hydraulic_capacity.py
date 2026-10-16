"""
Hydraulic Runoff & Culvert Flow Capacity Engine.
Implements storm peak runoff estimation (Rational/TR-55 method) and
Manning's culvert barrel hydraulic flow capacity under peak discharge events.
"""

import numpy as np
from typing import Dict, Any, Tuple


def calculate_peak_storm_runoff(
    catchment_area_km2: float,
    rainfall_intensity_mm_hr: float = 65.0,
    runoff_coefficient_c: float = 0.55
) -> float:
    """
    Computes peak storm discharge Q_peak (m^3/s) using the Rational/TR-55 formulation.
    
    Q_peak = (C * I * A) / 3.6
    Parameters:
        catchment_area_km2: Drainage basin area in square kilometers.
        rainfall_intensity_mm_hr: Design storm intensity (e.g., 25-yr 2-hr storm event in mm/hr).
        runoff_coefficient_c: Dimensionless runoff coefficient (0.2 to 0.9 depending on soil saturation).
        
    Returns:
        Q_peak: Peak discharge in cubic meters per second (m^3/s).
    """
    q_peak = (runoff_coefficient_c * rainfall_intensity_mm_hr * catchment_area_km2) / 3.6
    return round(float(q_peak), 3)


def calculate_culvert_manning_capacity(
    pipe_diameter_m: float = 1.2,
    slope_channel_deg: float = 2.0,
    manning_n: float = 0.024
) -> float:
    """
    Computes full-flowing hydraulic capacity Q_cap (m^3/s) of a circular corrugated metal pipe (CMP)
    or concrete culvert using Manning's open/gravity channel equation:
    
    Q_cap = (1 / n) * A * R_h^(2/3) * S_0^(1/2)
    
    Parameters:
        pipe_diameter_m: Interior culvert diameter in meters (typical standard 0.9m to 2.4m).
        slope_channel_deg: Bed slope gradient in degrees (converted to longitudinal slope S_0).
        manning_n: Roughness coefficient (0.013 for smooth concrete, 0.024 for corrugated metal pipe).
        
    Returns:
        Q_capacity: Maximum conveyance capacity in cubic meters per second (m^3/s).
    """
    radius = pipe_diameter_m / 2.0
    area = np.pi * (radius ** 2)
    wetted_perimeter = np.pi * pipe_diameter_m
    hydraulic_radius = area / wetted_perimeter # D / 4 for circular pipe
    
    # Longitudinal slope S_0 = tan(slope_deg)
    s_0 = np.tan(np.radians(max(slope_channel_deg, 0.2)))
    
    q_cap = (1.0 / manning_n) * area * (hydraulic_radius ** (2.0 / 3.0)) * np.sqrt(s_0)
    return round(float(q_cap), 3)


def evaluate_hydraulic_surcharge(
    q_peak: float,
    q_capacity: float
) -> Dict[str, Any]:
    """
    Evaluates hydraulic surcharge ratio and overtopping hazard tier.
    """
    surcharge_ratio = q_peak / max(q_capacity, 0.01)
    
    if surcharge_ratio <= 0.8:
        status = "Adequate Freeboard"
        hazard_level = "Low"
        color = "#2ECC71" # Green
    elif surcharge_ratio <= 1.0:
        status = "Near Capacity"
        hazard_level = "Moderate"
        color = "#F1C40F" # Yellow
    elif surcharge_ratio <= 1.4:
        status = "Hydraulic Surcharge (Inlet Ponding)"
        hazard_level = "High"
        color = "#E67E22" # Orange
    else:
        status = "Severe Overtopping / Embankment Spill"
        hazard_level = "Critical"
        color = "#E74C3C" # Red
        
    return {
        "q_peak_m3s": q_peak,
        "q_capacity_m3s": q_capacity,
        "surcharge_ratio": round(surcharge_ratio, 2),
        "status": status,
        "hazard_level": hazard_level,
        "color": color
    }
