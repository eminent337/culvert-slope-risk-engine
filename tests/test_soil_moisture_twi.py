"""
Unit tests for the Beven-Kirkby Topographic Wetness Index and SMAP downscaling module.
"""

import pytest
import numpy as np
from src.soil_moisture_twi import (
    compute_topographic_wetness_index,
    downscale_smap_soil_moisture
)


def test_twi_scaling_with_slope():
    """Flat ground with high accumulation should yield significantly higher TWI than steep slopes."""
    accum = np.array([[1000, 1000]])
    # Slope 1: flat (2 degrees), Slope 2: steep (40 degrees)
    slopes_rad = np.radians(np.array([[2.0, 40.0]]))
    
    twi = compute_topographic_wetness_index(accum, slopes_rad, resolution=25.0)
    assert twi[0, 0] > twi[0, 1] + 2.0


def test_smap_downscaling_physical_bounds():
    """Downscaled soil moisture must strictly obey physical saturation boundaries [0.05, 1.0]."""
    twi_grid = np.random.uniform(2.0, 18.0, (20, 20))
    moisture = downscale_smap_soil_moisture(twi_grid, regional_moisture_fraction=0.70)
    
    assert np.all(moisture >= 0.05)
    assert np.all(moisture <= 1.0)
