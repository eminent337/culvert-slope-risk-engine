"""
Unit tests for the USDA Forest Service woody debris & sediment inlet plugging module.
"""

import pytest
from src.debris_hazard import (
    estimate_bankfull_channel_width,
    calculate_debris_jamming_probability
)


def test_bankfull_width_scaling():
    """Bankfull channel width must increase monotonically with catchment area."""
    w1 = estimate_bankfull_channel_width(0.1)
    w2 = estimate_bankfull_channel_width(1.0)
    w3 = estimate_bankfull_channel_width(5.0)
    
    assert w1 < w2 < w3


def test_debris_jamming_narrow_pipe_risk():
    """A narrow culvert (0.8m) on a wide channel should have much higher plugging probability than a wide culvert (2.5m)."""
    p_small, _ = calculate_debris_jamming_probability(pipe_diameter_m=0.8, catchment_area_km2=2.0, stream_slope_deg=3.0)
    p_large, _ = calculate_debris_jamming_probability(pipe_diameter_m=2.5, catchment_area_km2=2.0, stream_slope_deg=3.0)
    
    assert p_small > p_large
