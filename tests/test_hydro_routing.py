"""
Unit tests for the terrain gradient and D8 hydrological routing engine.
"""

import pytest
import numpy as np
from src.hydro_routing import (
    calculate_slope_and_aspect,
    calculate_d8_flow_direction,
    calculate_flow_accumulation
)


def test_slope_gradient_constant_inclined_plane():
    """An analytical inclined plane z = 0.5 * x should yield an exact constant slope."""
    res = 10.0
    # 50x50 grid with 50% slope in X direction (tan(theta) = 0.5 => theta approx 26.565 deg)
    x = np.linspace(0, 500, 50)
    y = np.linspace(0, 500, 50)
    X, Y = np.meshgrid(x, y)
    dem = 0.5 * X
    
    slope_deg, slope_rad = calculate_slope_and_aspect(dem, resolution=res)
    
    # Check interior cells (avoid boundary finite difference artifact)
    interior_slope = slope_deg[5:-5, 5:-5]
    expected_deg = np.degrees(np.arctan(0.5))
    
    assert np.allclose(interior_slope, expected_deg, atol=0.5)


def test_d8_routing_pyramid():
    """A pyramid surface with a peak at center should route outward in all 4 cardinal directions."""
    res = 10.0
    grid = np.zeros((11, 11))
    for r in range(11):
        for c in range(11):
            dist = max(abs(r - 5), abs(c - 5))
            grid[r, c] = 100.0 - 5.0 * dist
            
    flow_dir = calculate_d8_flow_direction(grid, resolution=res)
    
    # Cell immediately East of center (5, 6) should flow East (code 1)
    assert flow_dir[5, 6] == 1
    # Cell immediately South of center (6, 5) should flow South (code 4)
    assert flow_dir[6, 5] == 4
    # Cell immediately West of center (5, 4) should flow West (code 16)
    assert flow_dir[5, 4] == 16
    # Cell immediately North of center (4, 5) should flow North (code 64)
    assert flow_dir[4, 5] == 64


def test_flow_accumulation_monotonicity():
    """Flow accumulation must be >= 1 for every valid cell."""
    res = 25.0
    # Simple tilted plane flowing towards South-East
    r = np.arange(20)[:, None]
    c = np.arange(20)[None, :]
    dem = 200.0 - 2.0 * r - 2.0 * c
    
    flow_dir = calculate_d8_flow_direction(dem, resolution=res)
    accum_cells, accum_km2 = calculate_flow_accumulation(dem, flow_dir, resolution=res)
    
    assert np.all(accum_cells >= 1)
    assert accum_cells[-1, -1] > 1 # Bottom corner must accumulate upstream flow
