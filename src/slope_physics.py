"""
Physics-Informed Geotechnical Slope Stability Engine.
Implements the Terzaghi-Coulomb Infinite Slope Factor of Safety (FS) model
coupled with hydrologic antecedent soil saturation and pore-water pressure.
"""

import numpy as np
from typing import Union, Dict, Any, Tuple, List


def calculate_factor_of_safety(
    slope_deg: Union[float, np.ndarray],
    cohesion_kpa: float = 8.0,
    friction_angle_deg: float = 32.0,
    soil_depth_m: float = 1.5,
    soil_unit_weight_kn_m3: float = 18.5,
    water_unit_weight_kn_m3: float = 9.81,
    saturation_ratio_m: Union[float, np.ndarray] = 0.5
) -> Union[float, np.ndarray]:
    """
    Computes the Factor of Safety (FS) for an infinite slope subjected to hydrologic saturation.
    
    Mathematical Formulation (Terzaghi-Coulomb with pore pressure):
        FS = [ c' + (gamma_s - m * gamma_w) * z * cos^2(beta) * tan(phi') ] /
             [ gamma_s * z * sin(beta) * cos(beta) ]
             
    Parameters:
        slope_deg: Ground slope gradient in degrees (beta).
        cohesion_kpa: Effective soil cohesion c' in kPa (kN/m^2).
        friction_angle_deg: Effective internal friction angle phi' in degrees.
        soil_depth_m: Depth of soil mantle down to slip plane z in meters.
        soil_unit_weight_kn_m3: Total unit weight of soil gamma_s in kN/m^3.
        water_unit_weight_kn_m3: Unit weight of water gamma_w in kN/m^3 (default 9.81).
        saturation_ratio_m: Hydrologic saturation ratio m = zw / z (0.0 = bone dry, 1.0 = fully saturated).
        
    Returns:
        FS: Factor of Safety (dimensionless ratio).
            FS > 1.5: Stable slope.
            1.0 <= FS <= 1.5: Conditionally stable / vulnerable to intense storm surges.
            FS < 1.0: Active slope failure / debris flow initiation.
    """
    beta = np.radians(np.maximum(slope_deg, 0.5)) # Guard against pure zero division
    phi = np.radians(friction_angle_deg)
    m = np.clip(saturation_ratio_m, 0.0, 1.0)
    
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    cos2_beta = cos_beta ** 2
    
    # Gravitational driving shear stress (tau_d = gamma_s * z * sin(beta) * cos(beta))
    driving_stress = soil_unit_weight_kn_m3 * soil_depth_m * sin_beta * cos_beta
    
    # Effective normal stress (sigma' = sigma - u = (gamma_s - m * gamma_w) * z * cos^2(beta))
    effective_normal_stress = (soil_unit_weight_kn_m3 - m * water_unit_weight_kn_m3) * soil_depth_m * cos2_beta
    effective_normal_stress = np.maximum(effective_normal_stress, 0.0) # Normal stress cannot be tensile
    
    # Resisting shear strength (tau_r = c' + sigma' * tan(phi))
    resisting_strength = cohesion_kpa + effective_normal_stress * np.tan(phi)
    
    # Factor of Safety
    fs = resisting_strength / np.maximum(driving_stress, 1e-4)
    
    # Cap flat slopes to practical ceiling
    fs = np.where(slope_deg <= 2.0, 10.0, fs)
    
    return fs


def compute_critical_saturation_threshold(
    slope_deg: float,
    cohesion_kpa: float = 8.0,
    friction_angle_deg: float = 32.0,
    soil_depth_m: float = 1.5,
    soil_unit_weight_kn_m3: float = 18.5,
    water_unit_weight_kn_m3: float = 9.81
) -> float:
    """
    Analytically solves for the critical hydrologic saturation ratio m_crit where FS = 1.0.
    If m_crit > 1.0: Slope remains stable even when 100% saturated.
    If m_crit <= 0.0: Slope is unconditionally unstable even when completely dry.
    """
    beta = np.radians(max(slope_deg, 0.5))
    phi = np.radians(friction_angle_deg)
    
    # At FS = 1.0:
    # gamma_s * z * sin(beta) * cos(beta) = c' + (gamma_s - m_crit * gamma_w) * z * cos^2(beta) * tan(phi)
    # Solving for m_crit:
    tan_phi = np.tan(phi)
    cos2_beta = np.cos(beta) ** 2
    sin_cos_beta = np.sin(beta) * np.cos(beta)
    
    denominator = water_unit_weight_kn_m3 * soil_depth_m * cos2_beta * tan_phi
    
    if denominator <= 1e-6:
        return 1.0
        
    numerator = cohesion_kpa + (soil_unit_weight_kn_m3 * soil_depth_m * cos2_beta * tan_phi) - (soil_unit_weight_kn_m3 * soil_depth_m * sin_cos_beta)
    
    m_crit = numerator / denominator
    return float(m_crit)


def evaluate_moisture_sensitivity_curve(
    slope_deg: float,
    cohesion_kpa: float = 8.0,
    friction_angle_deg: float = 32.0,
    soil_depth_m: float = 1.5,
    steps: int = 21
) -> Dict[str, Any]:
    """
    Generates a full sensitivity response curve of FS across saturation levels from m = 0.0 to 1.0.
    """
    m_values = np.linspace(0.0, 1.0, steps)
    fs_values = [
        float(calculate_factor_of_safety(slope_deg, cohesion_kpa, friction_angle_deg, soil_depth_m, saturation_ratio_m=m))
        for m in m_values
    ]
    
    m_crit = compute_critical_saturation_threshold(slope_deg, cohesion_kpa, friction_angle_deg, soil_depth_m)
    
    return {
        "slope_deg": slope_deg,
        "m_values": m_values.tolist(),
        "fs_values": [round(val, 3) for val in fs_values],
        "m_crit": round(m_crit, 3),
        "fails_under_saturation": m_crit <= 1.0
    }


def classify_slope_hazard(fs: float) -> Tuple[str, str]:
    """
    Returns hazard category and color hex code.
    """
    if fs >= 1.5:
        return "Stable", "#2ECC71"           # Green
    elif fs >= 1.2:
        return "Marginally Stable", "#F1C40F" # Yellow
    elif fs >= 1.0:
        return "Vulnerable", "#E67E22"        # Orange
    else:
        return "Failure Hazard", "#E74C3C"    # Red
