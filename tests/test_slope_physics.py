"""
Unit tests for the physics-informed geotechnical slope stability model.
"""

import pytest
import numpy as np
from src.slope_physics import (
    calculate_factor_of_safety,
    compute_critical_saturation_threshold,
    evaluate_moisture_sensitivity_curve,
    classify_slope_hazard
)


def test_factor_of_safety_flat_slope():
    """Flat ground should exhibit high stability (capped at 10.0)."""
    fs = calculate_factor_of_safety(slope_deg=1.0, saturation_ratio_m=0.5)
    assert fs >= 9.9


def test_factor_of_safety_degradation_with_saturation():
    """Factor of Safety must monotonically decrease as saturation ratio m increases."""
    slope = 32.0
    fs_dry = calculate_factor_of_safety(slope_deg=slope, saturation_ratio_m=0.0)
    fs_half = calculate_factor_of_safety(slope_deg=slope, saturation_ratio_m=0.5)
    fs_sat = calculate_factor_of_safety(slope_deg=slope, saturation_ratio_m=1.0)
    
    assert fs_dry > fs_half > fs_sat
    # Dry slope at friction angle 32 deg should be near 1.0 or higher with cohesion
    assert fs_dry > 1.0


def test_critical_saturation_threshold():
    """Analytical m_crit should predict the exact saturation where FS drops to 1.0."""
    slope = 30.0
    m_crit = compute_critical_saturation_threshold(slope_deg=slope)
    
    # If m_crit is within [0, 1], evaluating FS at m_crit should yield approximately 1.0
    if 0.0 <= m_crit <= 1.0:
        fs_at_crit = calculate_factor_of_safety(slope_deg=slope, saturation_ratio_m=m_crit)
        assert pytest.approx(fs_at_crit, rel=1e-2) == 1.0


def test_vectorized_slope_physics():
    """Verify function behaves consistently when passed NumPy arrays."""
    slopes = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
    fs_array = calculate_factor_of_safety(slope_deg=slopes, saturation_ratio_m=0.6)
    
    assert isinstance(fs_array, np.ndarray)
    assert len(fs_array) == len(slopes)
    # Steeper slopes should have lower factor of safety
    assert np.all(np.diff(fs_array) < 0)


def test_hazard_classification():
    """Verify categorical hazard binning."""
    assert classify_slope_hazard(1.8)[0] == "Stable"
    assert classify_slope_hazard(1.3)[0] == "Marginally Stable"
    assert classify_slope_hazard(1.1)[0] == "Vulnerable"
    assert classify_slope_hazard(0.8)[0] == "Failure Hazard"
