"""
Road-Stream Spatial Intersection & Culvert Catchment Delineation Engine.
Identifies physical infrastructure crossing coordinates and isolates the upstream
contributing drainage catchment and terrain gradients for each culvert candidate.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, LineString
from typing import List, Dict, Any, Tuple


# Inverted D8 offsets: Which upstream neighbor flows into (r, c)?
# If neighbor has direction code, it flows into current cell.
UPSTREAM_OFFSETS = [
    (0, -1, 1),     # Neighbor to West with flow_dir=East (1)
    (-1, -1, 2),    # Neighbor to North-West with flow_dir=SE (2)
    (-1, 0, 4),     # Neighbor to North with flow_dir=South (4)
    (-1, 1, 8),     # Neighbor to North-East with flow_dir=SW (8)
    (0, 1, 16),     # Neighbor to East with flow_dir=West (16)
    (1, 1, 32),     # Neighbor to South-East with flow_dir=NW (32)
    (1, 0, 64),     # Neighbor to South with flow_dir=North (64)
    (1, -1, 128)    # Neighbor to South-West with flow_dir=NE (128)
]


def detect_culvert_crossings(
    roads_gdf: gpd.GeoDataFrame,
    streams_gdf: gpd.GeoDataFrame,
    cluster_tolerance_m: float = 20.0
) -> gpd.GeoDataFrame:
    """
    Performs spatial geometric intersection between road vector lines and hydrographic stream lines.
    Extracts crossing points and deduplicates them using spatial distance clustering.
    """
    # Ensure CRS synchronization
    if roads_gdf.crs != streams_gdf.crs:
        roads_gdf = roads_gdf.to_crs(streams_gdf.crs)
        
    crossing_records: List[Dict[str, Any]] = []
    
    for _, road in roads_gdf.iterrows():
        for _, stream in streams_gdf.iterrows():
            if road.geometry.intersects(stream.geometry):
                inter = road.geometry.intersection(stream.geometry)
                
                # Intersections can be Point, MultiPoint, or GeometryCollection
                points = []
                if isinstance(inter, Point):
                    points.append(inter)
                elif isinstance(inter, MultiPoint):
                    points.extend(list(inter.geoms))
                elif hasattr(inter, "geoms"):
                    for g in inter.geoms:
                        if isinstance(g, Point):
                            points.append(g)
                            
                for pt in points:
                    crossing_records.append({
                        "road_id": road.get("road_id", "UNKNOWN"),
                        "road_name": road.get("name", "Unnamed Road"),
                        "road_type": road.get("type", "General"),
                        "stream_id": stream.get("reach_id", "STREAM"),
                        "geometry": pt
                    })
                    
    if not crossing_records:
        return gpd.GeoDataFrame(columns=["culvert_id", "geometry"], crs=roads_gdf.crs)
        
    raw_crossings = gpd.GeoDataFrame(crossing_records, crs=roads_gdf.crs)
    
    # Deduplicate points that fall within cluster_tolerance_m
    deduped = []
    used_indices = set()
    
    for i, row in raw_crossings.iterrows():
        if i in used_indices:
            continue
            
        pt = row.geometry
        # Find neighbors within tolerance
        distances = raw_crossings.geometry.distance(pt)
        close_mask = distances <= cluster_tolerance_m
        close_indices = close_mask[close_mask].index
        
        used_indices.update(close_indices)
        
        # Centroid of clustered points
        cluster_pts = raw_crossings.loc[close_indices, 'geometry']
        mean_x = cluster_pts.x.mean()
        mean_y = cluster_pts.y.mean()
        
        cid = f"CULV-{len(deduped)+1:03d}"
        deduped.append({
            "culvert_id": cid,
            "road_id": row["road_id"],
            "road_name": row["road_name"],
            "road_type": row["road_type"],
            "stream_id": row["stream_id"],
            "utm_easting": mean_x,
            "utm_northing": mean_y,
            "geometry": Point(mean_x, mean_y)
        })
        
    return gpd.GeoDataFrame(deduped, crs=roads_gdf.crs)


def delineate_catchment_mask(
    pour_point: Point,
    dem: np.ndarray,
    flow_dir: np.ndarray,
    transform: Affine
) -> np.ndarray:
    """
    Back-traces the upstream contributing watershed from a specific pour point (culvert location)
    using the D8 flow direction matrix. Returns a boolean mask of the contributing basin.
    """
    rows, cols = dem.shape
    # Convert point coordinates (X, Y) to raster row, col
    col, row = ~transform * (pour_point.x, pour_point.y)
    r_start, c_start = int(round(row)), int(round(col))
    
    catchment = np.zeros((rows, cols), dtype=bool)
    
    # Bounds check
    if not (0 <= r_start < rows and 0 <= c_start < cols):
        return catchment
        
    # Queue for breadth-first / depth-first watershed back-tracing
    queue = [(r_start, c_start)]
    catchment[r_start, c_start] = True
    
    while queue:
        curr_r, curr_c = queue.pop()
        
        # Check all 8 neighbors: does neighbor flow INTO curr_r, curr_c?
        for dr, dc, required_code in UPSTREAM_OFFSETS:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if not catchment[nr, nc] and flow_dir[nr, nc] == required_code:
                    catchment[nr, nc] = True
                    queue.append((nr, nc))
                    
    return catchment


def extract_culvert_catchment_attributes(
    crossings_gdf: gpd.GeoDataFrame,
    dem: np.ndarray,
    slope_deg: np.ndarray,
    flow_dir: np.ndarray,
    accum_area_km2: np.ndarray,
    transform: Affine,
    resolution: float
) -> gpd.GeoDataFrame:
    """
    Attaches detailed upstream terrain and hydrologic catchment statistics to each culvert crossing.
    """
    rows, cols = dem.shape
    enhanced_records = []
    
    for _, crossing in crossings_gdf.iterrows():
        rec = dict(crossing)
        pt = crossing.geometry
        
        # Identify pour point pixel
        c_float, r_float = ~transform * (pt.x, pt.y)
        r = int(np.clip(round(r_float), 0, rows - 1))
        c = int(np.clip(round(c_float), 0, cols - 1))
        
        # Local culvert elevation and flow accumulation
        culvert_elev = float(dem[r, c])
        local_accum = float(accum_area_km2[r, c])
        
        # Delineate upstream basin
        basin_mask = delineate_catchment_mask(pt, dem, flow_dir, transform)
        basin_cell_count = int(np.sum(basin_mask))
        
        if basin_cell_count > 1:
            basin_slopes = slope_deg[basin_mask]
            basin_elevs = dem[basin_mask]
            
            drainage_area_km2 = (basin_cell_count * resolution * resolution) / 1e6
            mean_slope = float(np.mean(basin_slopes))
            p90_slope = float(np.percentile(basin_slopes, 90))
            max_slope = float(np.max(basin_slopes))
            relief_m = float(np.max(basin_elevs) - np.min(basin_elevs))
        else:
            # Fallback to local 5x5 window if basin back-tracing hits edge
            r_min, r_max = max(0, r-2), min(rows, r+3)
            c_min, c_max = max(0, c-2), min(cols, c+3)
            local_slopes = slope_deg[r_min:r_max, c_min:c_max]
            
            drainage_area_km2 = max(local_accum, 0.05)
            mean_slope = float(np.mean(local_slopes))
            p90_slope = float(np.percentile(local_slopes, 90))
            max_slope = float(np.max(local_slopes))
            relief_m = 50.0
            
        rec["elevation_m"] = culvert_elev
        rec["drainage_area_km2"] = round(drainage_area_km2, 3)
        rec["mean_upslope_deg"] = round(mean_slope, 1)
        rec["p90_upslope_deg"] = round(p90_slope, 1)
        rec["max_upslope_deg"] = round(max_slope, 1)
        rec["basin_relief_m"] = round(relief_m, 1)
        
        enhanced_records.append(rec)
        
    return gpd.GeoDataFrame(enhanced_records, crs=crossings_gdf.crs)
