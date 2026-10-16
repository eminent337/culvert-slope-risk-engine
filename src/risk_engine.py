"""
Integrated Multi-Hazard Culvert Risk Scoring & Prioritization Engine.
Synthesizes geotechnical slope stability (FS), hydraulic surcharge (HR),
and roadway infrastructure consequence into an operational decision index.
"""

import numpy as np
import geopandas as gpd
from typing import Dict, Any, List
from .slope_physics import calculate_factor_of_safety, compute_critical_saturation_threshold
from .hydraulic_capacity import calculate_peak_storm_runoff, calculate_culvert_manning_capacity, evaluate_hydraulic_surcharge


# Road consequence weights based on traffic speed and strategic importance
ROAD_CRITICALITY_WEIGHTS = {
    "Primary Highway": 1.0,
    "Rural Collector": 0.75,
    "Secondary Unpaved": 0.50,
    "Tertiary Mountain Road": 0.35,
    "General": 0.50
}


def assess_culvert_network_risk(
    culverts_gdf: gpd.GeoDataFrame,
    antecedent_saturation_m: float = 0.65,
    design_rainfall_intensity_mm_hr: float = 70.0,
    default_pipe_diameter_m: float = 1.2,
    soil_cohesion_kpa: float = 8.0,
    soil_friction_angle_deg: float = 32.0,
    soil_depth_m: float = 1.5
) -> gpd.GeoDataFrame:
    """
    Evaluates coupled geotechnical and hydraulic risk across all detected culvert crossings.
    
    Parameters:
        culverts_gdf: GeoDataFrame containing culvert crossing points and upstream catchment attributes.
        antecedent_saturation_m: Pre-storm soil moisture saturation ratio (0.0 to 1.0).
        design_rainfall_intensity_mm_hr: Design storm rainfall intensity in mm/hr.
        default_pipe_diameter_m: Assumed culvert pipe diameter in meters.
        soil_cohesion_kpa: Effective soil cohesion c' in kPa.
        soil_friction_angle_deg: Soil internal friction angle phi' in degrees.
        soil_depth_m: Soil mantle thickness down to shear plane in meters.
        
    Returns:
        Enhanced GeoDataFrame ranked by Composite Vulnerability Index (CVI).
    """
    assessed_records: List[Dict[str, Any]] = []
    
    for _, row in culverts_gdf.iterrows():
        rec = dict(row)
        
        slope_deg = float(rec.get("p90_upslope_deg", rec.get("mean_upslope_deg", 15.0)))
        area_km2 = float(rec.get("drainage_area_km2", 0.5))
        road_type = str(rec.get("road_type", "General"))
        
        # 1. Geotechnical Slope Stability under Antecedent Moisture
        fs = float(calculate_factor_of_safety(
            slope_deg=slope_deg,
            cohesion_kpa=soil_cohesion_kpa,
            friction_angle_deg=soil_friction_angle_deg,
            soil_depth_m=soil_depth_m,
            saturation_ratio_m=antecedent_saturation_m
        ))
        
        m_crit = float(compute_critical_saturation_threshold(
            slope_deg=slope_deg,
            cohesion_kpa=soil_cohesion_kpa,
            friction_angle_deg=soil_friction_angle_deg,
            soil_depth_m=soil_depth_m
        ))
        
        # 2. Hydraulic Peak Runoff & Surcharge
        # Adjust runoff coefficient higher if soil is already saturated (preconditioned)
        c_runoff = 0.45 + (0.35 * antecedent_saturation_m)
        q_peak = calculate_peak_storm_runoff(
            catchment_area_km2=area_km2,
            rainfall_intensity_mm_hr=design_rainfall_intensity_mm_hr,
            runoff_coefficient_c=c_runoff
        )
        
        q_cap = calculate_culvert_manning_capacity(
            pipe_diameter_m=default_pipe_diameter_m,
            slope_channel_deg=max(slope_deg * 0.15, 0.5) # Culvert bed slope typically shallower than hillslope
        )
        
        hyd_eval = evaluate_hydraulic_surcharge(q_peak, q_cap)
        hr = hyd_eval["surcharge_ratio"]
        
        # 3. Normalized Component Hazard Scores (0 to 100)
        # Geotechnical hazard: High when FS < 1.0, zero when FS >= 2.0
        if fs <= 1.0:
            score_geotech = 100.0
        elif fs >= 2.0:
            score_geotech = 0.0
        else:
            score_geotech = (2.0 - fs) * 100.0 # Linear scaling between 1.0 and 2.0
            
        # Hydraulic surcharge hazard: High when HR >= 1.5
        score_hydraulic = min(100.0, (hr / 1.5) * 100.0)
        
        # Road consequence weight (0.35 to 1.0)
        road_weight = ROAD_CRITICALITY_WEIGHTS.get(road_type, 0.50)
        
        # 4. Composite Vulnerability Index (CVI)
        # Weighted composite + Embankment Blowout Penalty
        # If hydraulic surcharge (HR > 1.0) coincides with slope instability (FS < 1.2),
        # an embankment blowout multiplier is applied!
        blowout_penalty = 1.35 if (hr >= 1.0 and fs <= 1.25) else 1.0
        
        cvi = ((0.50 * score_geotech) + (0.50 * score_hydraulic)) * road_weight * blowout_penalty
        cvi = round(min(100.0, cvi), 1)
        
        # Risk Category Assignment
        if cvi >= 75.0 or (hr >= 1.5 and fs <= 1.1):
            risk_tier = "CRITICAL (BLOWOUT HAZARD)"
            risk_color = "#E74C3C" # Bright Red
        elif cvi >= 50.0:
            risk_tier = "HIGH RISK"
            risk_color = "#E67E22" # Orange
        elif cvi >= 25.0:
            risk_tier = "MODERATE RISK"
            risk_color = "#F1C40F" # Amber / Yellow
        else:
            risk_tier = "LOW RISK"
            risk_color = "#2ECC71" # Green
            
        rec["factor_of_safety"] = round(fs, 2)
        rec["critical_saturation_mcrit"] = round(m_crit, 2)
        rec["q_peak_m3s"] = q_peak
        rec["q_capacity_m3s"] = q_cap
        rec["hydraulic_surcharge_ratio"] = hr
        rec["hydraulic_status"] = hyd_eval["status"]
        rec["composite_vulnerability_index"] = cvi
        rec["risk_tier"] = risk_tier
        rec["risk_color"] = risk_color
        rec["blowout_hazard"] = bool(blowout_penalty > 1.0)
        
        assessed_records.append(rec)
        
    result_gdf = gpd.GeoDataFrame(assessed_records, crs=culverts_gdf.crs)
    # Sort descending by risk score
    result_gdf = result_gdf.sort_values(by="composite_vulnerability_index", ascending=False).reset_index(drop=True)
    return result_gdf
