"""
Unit tests for the hydraulic runoff and culvert capacity engine.
"""

import pytest
from src.hydraulic_capacity import (
    calculate_peak_storm_runoff,
    calculate_culvert_manning_capacity,
    evaluate_hydraulic_surcharge
)


def test_peak_runoff_scaling():
    """Peak runoff should scale linearly with catchment area."""
    q1 = calculate_peak_storm_runoff(catchment_area_km2=1.0, rainfall_intensity_mm_hr=50.0)
    q2 = calculate_peak_storm_runoff(catchment_area_km2=2.0, rainfall_intensity_mm_hr=50.0)
    assert pytest.approx(q2, rel=1e-3) == 2.0 * q1


def test_manning_capacity_scaling_with_diameter():
    """Culvert capacity should scale strongly with diameter (D^(8/3))."""
    cap_small = calculate_culvert_manning_capacity(pipe_diameter_m=1.0, slope_channel_deg=2.0)
    cap_large = calculate_culvert_manning_capacity(pipe_diameter_m=2.0, slope_channel_deg=2.0)
    
    assert cap_large > 4.0 * cap_small


def test_hydraulic_surcharge_tiers():
    """Verify overtopping hazard categorization."""
    eval_safe = evaluate_hydraulic_surcharge(q_peak=2.0, q_capacity=5.0)
    assert eval_safe["hazard_level"] == "Low"
    assert eval_safe["surcharge_ratio"] == 0.4
    
    eval_overtopping = evaluate_hydraulic_surcharge(q_peak=8.0, q_capacity=4.0)
    assert eval_overtopping["hazard_level"] == "Critical"
    assert eval_overtopping["surcharge_ratio"] == 2.0
