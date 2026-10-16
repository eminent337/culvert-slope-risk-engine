"""
Unit tests for the Sentinel-1 InSAR kinematic observational prior module.
"""

import pytest
import numpy as np
from src.insar_prior import (
    generate_synthetic_sentinel1_insar_field,
    update_factor_of_safety_with_insar
)


def test_insar_stationary_stability_preservation():
    """When InSAR LOS motion is below the geodetic noise floor (e.g. 2 mm/yr), FS should remain unchanged."""
    fs_prior = 1.65
    fs_updated, meta = update_factor_of_safety_with_insar(fs_prior, v_los_mm_yr=-2.0)
    
    assert fs_updated == fs_prior
    assert meta["kinematic_status"] == "Stationary (Stable Crust)"


def test_insar_active_creep_degradation():
    """When InSAR measures severe downhill displacement (e.g. -25 mm/yr), FS must degrade significantly."""
    fs_prior = 1.40
    fs_updated, meta = update_factor_of_safety_with_insar(fs_prior, v_los_mm_yr=-25.0)
    
    assert fs_updated < fs_prior
    assert meta["activity_level"] == "Severe"
    assert meta["kinematic_degradation_pct"] > 25.0
