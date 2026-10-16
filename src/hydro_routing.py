"""
Topographic Analysis and Hydrological Routing Engine.
Implements DEM hydro-conditioning, finite-difference slope gradients,
D8 flow direction, flow accumulation, and hydrographic stream network vectorization.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import uniform_filter
import geopandas as gpd
from shapely.geometry import LineString
from typing import Tuple, Dict, Any, List


# D8 Neighbor Offsets: (dy, dx) indexed clockwise from East
# 1: E, 2: SE, 4: S, 8: SW, 16: W, 32: NW, 64: N, 128: NE
D8_DIRECTIONS = [
    (0, 1, 1),     # East
    (1, 1, 2),     # South-East
    (1, 0, 4),     # South
    (1, -1, 8),    # South-West
    (0, -1, 16),   # West
    (-1, -1, 32),  # North-West
    (-1, 0, 64),   # North
    (-1, 1, 128)   # North-East
]


def fill_depressions(dem: np.ndarray, epsilon: float = 0.01, max_iter: int = 50) -> np.ndarray:
    """
    Hydro-conditions the DEM by carving/raising small depressions and sinks
    so flow routing does not terminate prematurely in artificial pits.
    Uses an efficient morphological sink-raising filter.
    """
    filled = np.copy(dem)
    rows, cols = filled.shape
    
    # Iterative priority sink filling
    for _ in range(max_iter):
        shifted = np.copy(filled)
        # Check minimum of 8 neighbors
        min_neighbor = np.full_like(filled, np.inf)
        for dy, dx, _ in D8_DIRECTIONS:
            # Shifted view with edge handling
            padded = np.pad(filled, 1, mode='edge')
            shifted_n = padded[1+dy:1+dy+rows, 1+dx:1+dx+cols]
            min_neighbor = np.minimum(min_neighbor, shifted_n)
            
        # Raise cells that are lower than all surrounding neighbors
        pits = (filled < min_neighbor)
        if not np.any(pits):
            break
        filled[pits] = min_neighbor[pits] + epsilon
        
    return filled


def calculate_slope_and_aspect(dem: np.ndarray, resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates topographic slope gradient (in degrees and radians) using
    second-order centered finite differences (Horn's algorithm).
    
    Returns:
        slope_degrees: Array of slope angles in degrees (0 to 90).
        slope_radians: Array of slope angles in radians (0 to pi/2).
    """
    # 2nd order centered differences for gradient in Y (North-South) and X (West-East)
    # Using np.gradient with spatial resolution spacing
    grad_y, grad_x = np.gradient(dem, resolution, resolution)
    
    # Slope tangent
    slope_tan = np.sqrt(grad_x**2 + grad_y**2)
    slope_radians = np.arctan(slope_tan)
    slope_degrees = np.degrees(slope_radians)
    
    return slope_degrees, slope_radians


def calculate_d8_flow_direction(dem: np.ndarray, resolution: float) -> np.ndarray:
    """
    Computes the D8 steepest descent flow direction matrix for every cell.
    Returns:
        flow_dir: Array containing D8 direction codes (1, 2, 4, 8, 16, 32, 64, 128, or 0 for flat).
    """
    rows, cols = dem.shape
    flow_dir = np.zeros((rows, cols), dtype=np.uint8)
    max_gradient = np.zeros((rows, cols), dtype=np.float32)
    
    padded = np.pad(dem, 1, mode='edge')
    
    for dy, dx, code in D8_DIRECTIONS:
        shifted = padded[1+dy:1+dy+rows, 1+dx:1+dx+cols]
        # Distance: orthogonal = resolution, diagonal = resolution * sqrt(2)
        dist = resolution * np.sqrt(dx**2 + dy**2)
        drop = (dem - shifted) / dist
        
        steeper = (drop > max_gradient)
        flow_dir[steeper] = code
        max_gradient[steeper] = drop[steeper]
        
    return flow_dir


def calculate_flow_accumulation(dem: np.ndarray, flow_dir: np.ndarray, resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes total contributing upslope area for every cell using topological elevation ordering.
    
    Returns:
        accum_cells: Contributing upstream cell count.
        accum_area_km2: Contributing area in square kilometers.
    """
    rows, cols = dem.shape
    accum = np.ones((rows, cols), dtype=np.int32) # Each cell contributes 1 (itself)
    
    # Sort cells from highest elevation to lowest elevation (topological sort for DAG)
    flat_indices = np.argsort(-dem.ravel())
    
    # Map from D8 code to (dy, dx)
    code_to_offset = {code: (dy, dx) for dy, dx, code in D8_DIRECTIONS}
    
    for idx in flat_indices:
        r = idx // cols
        c = idx % cols
        code = flow_dir[r, c]
        if code in code_to_offset:
            dy, dx = code_to_offset[code]
            nr, nc = r + dy, c + dx
            if 0 <= nr < rows and 0 <= nc < cols:
                accum[nr, nc] += accum[r, c]
                
    cell_area_km2 = (resolution * resolution) / 1e6
    accum_area_km2 = accum * cell_area_km2
    
    return accum, accum_area_km2


def extract_stream_network(
    accum_area_km2: np.ndarray,
    transform: Affine,
    crs: str,
    threshold_km2: float = 0.5
) -> gpd.GeoDataFrame:
    """
    Extracts vectorized hydrographic stream channels where contributing area >= threshold_km2.
    Traces continuous channel segments into GeoPandas LineString geometries.
    """
    rows, cols = accum_area_km2.shape
    is_stream = accum_area_km2 >= threshold_km2
    
    stream_segments: List[Dict[str, Any]] = []
    visited = np.zeros_like(is_stream, dtype=bool)
    
    # Connect stream cells into vector LineStrings
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if is_stream[r, c] and not visited[r, c]:
                # Start tracing a stream reach
                reach_coords = []
                curr_r, curr_c = r, c
                
                while 0 <= curr_r < rows and 0 <= curr_c < cols and is_stream[curr_r, curr_c] and not visited[curr_r, curr_c]:
                    visited[curr_r, curr_c] = True
                    # Affine transform to spatial coordinates (X, Y)
                    x, y = rasterio.transform.xy(transform, curr_r, curr_c)
                    reach_coords.append((x, y))
                    
                    # Look for downhill neighbor with maximum accumulation
                    best_acc = -1
                    next_r, next_c = -1, -1
                    for dy, dx, _ in D8_DIRECTIONS:
                        nr, nc = curr_r + dy, curr_c + dx
                        if 0 <= nr < rows and 0 <= nc < cols and is_stream[nr, nc]:
                            if accum_area_km2[nr, nc] > best_acc and not visited[nr, nc]:
                                best_acc = accum_area_km2[nr, nc]
                                next_r, next_c = nr, nc
                                
                    if next_r != -1:
                        curr_r, curr_c = next_r, next_c
                    else:
                        break
                        
                if len(reach_coords) >= 2:
                    stream_segments.append({
                        "reach_id": f"STRM_{len(stream_segments)+1:03d}",
                        "max_accum_km2": float(np.max(accum_area_km2[r, c])),
                        "geometry": LineString(reach_coords)
                    })
                    
    gdf = gpd.GeoDataFrame(stream_segments, crs=crs)
    return gdf
