"""
Integrated Advanced Multi-Hazard Culvert Risk Scoring & Prioritization Engine (v2.0).
Synthesizes:
1. Spatially distributed 2D soil moisture (Beven-Kirkby TWI / SMAP downscaling).
2. Geotechnical slope stability (Terzaghi-Coulomb FS).
3. Sentinel-1 multi-temporal InSAR ground deformation velocity priors (mm/year).
4. USDA Forest Service woody debris & sediment inlet plugging probability (GRAIP).
5. Effective hydraulic conveyance capacity and road consequence weights.
"""

import numpy as np
import geopandas as gpd
from typing import Dict, Any, List, Optional
from .slope_physics import calculate_factor_of_safety, compute_critical_saturation_threshold
from .hydraulic_capacity import calculate_peak_storm_runoff, calculate_culvert_manning_capacity, evaluate_hydraulic_surcharge
from .insar_prior import update_factor_of_safety_with_insar
from .debris_hazard import calculate_debris_jamming_probability


# Road consequence weights based on functional classification and emergency corridor criticality
ROAD_CRITICALITY_WEIGHTS = {
    "Primary Highway": 1.0,
    "Rural Collector": 0.75,
    "Secondary Unpaved": 0.50,
    "Tertiary Mountain Road": 0.35,
    "General": 0.50
}


def assess_culvert_network_risk_advanced(
    culverts_gdf: gpd.GeoDataFrame,
    saturation_raster_m: Optional[np.ndarray] = None,
    insar_velocity_raster: Optional[np.ndarray] = None,
    dem_transform = None,
    regional_saturation_scalar: float = 0.65,
    design_rainfall_intensity_mm_hr: float = 75.0,
    default_pipe_diameter_m: float = 1.2,
    soil_cohesion_kpa: float = 8.0,
    soil_friction_angle_deg: float = 32.0,
    soil_depth_m: float = 1.5,
    enable_insar_coupling: bool = True,
    enable_debris_plugging: bool = True
) -> gpd.GeoDataFrame:
    """
    Advanced multi-hazard culvert risk assessment integrating InSAR velocity priors,
    spatially distributed soil saturation, and USDA debris plugging models.
    """
    assessed_records: List[Dict[str, Any]] = []
    
    for _, row in culverts_gdf.iterrows():
        rec = dict(row)
        pt = row.geometry
        
        slope_deg = float(rec.get("p90_upslope_deg", rec.get("mean_upslope_deg", 15.0)))
        area_km2 = float(rec.get("drainage_area_km2", 0.5))
        road_type = str(rec.get("road_type", "General"))
        
        # 1. Local Soil Saturation (From 2D TWI raster if provided, else fallback to regional scalar)
        if saturation_raster_m is not None and dem_transform is not None:
            c_float, r_float = ~dem_transform * (pt.x, pt.y)
            r = int(np.clip(round(r_float), 0, saturation_raster_m.shape[0] - 1))
            c = int(np.clip(round(c_float), 0, saturation_raster_m.shape[1] - 1))
            local_saturation = float(saturation_raster_m[r, c])
        else:
            local_saturation = regional_saturation_scalar
            
        # 2. Prior Geotechnical Factor of Safety (Terzaghi-Coulomb)
        fs_prior = float(calculate_factor_of_safety(
            slope_deg=slope_deg,
            cohesion_kpa=soil_cohesion_kpa,
            friction_angle_deg=soil_friction_angle_deg,
            soil_depth_m=soil_depth_m,
            saturation_ratio_m=local_saturation
        ))
        
        m_crit = float(compute_critical_saturation_threshold(
            slope_deg=slope_deg,
            cohesion_kpa=soil_cohesion_kpa,
            friction_angle_deg=soil_friction_angle_deg,
            soil_depth_m=soil_depth_m
        ))
        
        # 3. Sentinel-1 InSAR Observational Updating
        v_los_mm_yr = 0.0
        insar_status = "Not Monitored"
        fs_final = fs_prior
        
        if enable_insar_coupling and insar_velocity_raster is not None and dem_transform is not None:
            c_float, r_float = ~dem_transform * (pt.x, pt.y)
            r = int(np.clip(round(r_float), 0, insar_velocity_raster.shape[0] - 1))
            c = int(np.clip(round(c_float), 0, insar_velocity_raster.shape[1] - 1))
            v_los_mm_yr = float(insar_velocity_raster[r, c])
            
            fs_final, insar_meta = update_factor_of_safety_with_insar(fs_prior, v_los_mm_yr)
            insar_status = insar_meta["kinematic_status"]
        else:
            fs_final = fs_prior
            
        # 4. Debris Plugging Probability (USDA GRAIP model)
        p_plugging = 0.0
        retention_factor = 1.0
        debris_tier = "Not Assessed"
        
        if enable_debris_plugging:
            p_plugging, debris_diag = calculate_debris_jamming_probability(
                pipe_diameter_m=default_pipe_diameter_m,
                catchment_area_km2=area_km2,
                stream_slope_deg=max(slope_deg * 0.15, 0.5)
            )
            retention_factor = debris_diag["capacity_retention_factor"]
            debris_tier = debris_diag["plugging_tier"]
            
        # 5. Hydraulic Runoff & Effective Conveyance Capacity
        c_runoff = 0.40 + (0.45 * local_saturation)
        q_peak = calculate_peak_storm_runoff(
            catchment_area_km2=area_km2,
            rainfall_intensity_mm_hr=design_rainfall_intensity_mm_hr,
            runoff_coefficient_c=c_runoff
        )
        
        q_raw_cap = calculate_culvert_manning_capacity(
            pipe_diameter_m=default_pipe_diameter_m,
            slope_channel_deg=max(slope_deg * 0.15, 0.5)
        )
        
        # Effective capacity reduced by woody debris / sediment clogging!
        q_effective_cap = round(float(q_raw_cap * retention_factor), 3)
        
        hyd_eval = evaluate_hydraulic_surcharge(q_peak, q_effective_cap)
        hr = hyd_eval["surcharge_ratio"]
        
        # 6. Multi-Component Scoring
        # Geotechnical Hazard (0 to 100)
        if fs_final <= 1.0:
            score_geotech = 100.0
        elif fs_final >= 2.0:
            score_geotech = 0.0
        else:
            score_geotech = (2.0 - fs_final) * 100.0
            
        # Hydraulic Overtopping Hazard (0 to 100)
        score_hydraulic = min(100.0, (hr / 1.5) * 100.0)
        
        # InSAR Kinematic Confirmation Multiplier (Active displacement accelerates failure)
        insar_multiplier = 1.0 + min(0.35, max(0.0, (abs(v_los_mm_yr) - 4.0) / 40.0))
        
        # Road consequence weight
        road_weight = ROAD_CRITICALITY_WEIGHTS.get(road_type, 0.50)
        
        # Embankment Blowout Multiplier: Severe Overtopping + Geotechnical Instability
        blowout_penalty = 1.40 if (hr >= 1.0 and fs_final <= 1.25) else 1.0
        
        # Composite Vulnerability Index
        cvi = ((0.45 * score_geotech) + (0.45 * score_hydraulic) + (0.10 * p_plugging * 100.0)) * road_weight * blowout_penalty * insar_multiplier
        cvi = round(min(100.0, cvi), 1)
        
        # Hazard Tier Classification
        if cvi >= 70.0 or (hr >= 1.4 and fs_final <= 1.15):
            risk_tier = "CRITICAL (BLOWOUT HAZARD)"
            risk_color = "#E74C3C" # Red
        elif cvi >= 45.0:
            risk_tier = "HIGH RISK"
            risk_color = "#E67E22" # Orange
        elif cvi >= 20.0:
            risk_tier = "MODERATE RISK"
            risk_color = "#F1C40F" # Yellow
        else:
            risk_tier = "LOW RISK"
            risk_color = "#2ECC71" # Green
            
        rec["local_saturation_m"] = round(local_saturation, 2)
        rec["factor_of_safety_prior"] = round(fs_prior, 2)
        rec["factor_of_safety_insar"] = round(fs_final, 2)
        rec["insar_velocity_los_mmyr"] = round(v_los_mm_yr, 1)
        rec["insar_status"] = insar_status
        rec["critical_saturation_mcrit"] = round(m_crit, 2)
        rec["debris_plugging_prob"] = round(p_plugging, 2)
        rec["debris_risk_tier"] = debris_tier
        rec["q_peak_m3s"] = q_peak
        rec["q_nominal_capacity_m3s"] = q_raw_cap
        rec["q_effective_capacity_m3s"] = q_effective_cap
        rec["hydraulic_surcharge_ratio"] = hr
        rec["hydraulic_status"] = hyd_eval["status"]
        rec["composite_vulnerability_index"] = cvi
        rec["risk_tier"] = risk_tier
        rec["risk_color"] = risk_color
        rec["blowout_hazard"] = bool(blowout_penalty > 1.0)
        
        assessed_records.append(rec)
        
    result_gdf = gpd.GeoDataFrame(assessed_records, crs=culverts_gdf.crs)
    result_gdf = result_gdf.sort_values(by="composite_vulnerability_index", ascending=False).reset_index(drop=True)
    return result_gdf
