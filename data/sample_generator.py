"""
Synthetic High-Fidelity Mountain Watershed and Road Network Generator.
Creates a georeferenced Digital Elevation Model (GeoTIFF) and vector road network (GeoJSON)
for realistic hydrological and geotechnical culvert risk modeling.
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import LineString
from pathlib import Path


def generate_mountain_dem(output_path: Path, width: int = 400, height: int = 400, resolution: float = 25.0):
    """
    Generates a realistic mountainous terrain with valleys, ridges, and a primary drainage channel.
    
    Parameters:
        output_path: Destination path for the GeoTIFF raster.
        width: Number of columns (pixels).
        height: Number of rows (pixels).
        resolution: Grid cell resolution in meters (e.g., 25m).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Coordinate grids in kilometers
    x = np.linspace(0, (width * resolution) / 1000.0, width)
    y = np.linspace(0, (height * resolution) / 1000.0, height)
    X, Y = np.meshgrid(x, y)
    
    # Base regional slope dipping from North-East to South-West
    regional_trend = 1200.0 - 0.05 * (X * 1000.0 + Y * 1000.0)
    
    # Mountain ridges and structural valleys
    ridge_1 = 350.0 * np.sin(2.0 * np.pi * X / 4.0) * np.cos(2.0 * np.pi * Y / 5.0)
    ridge_2 = 180.0 * np.cos(2.0 * np.pi * (X + Y) / 3.0)
    
    # Primary river canyon running diagonally through the landscape
    canyon_axis = Y - 0.7 * X - 1.5
    canyon_depth = 280.0 * np.exp(-(canyon_axis ** 2) / 1.8)
    
    # Tributary gullies cutting into the side slopes
    gullies = 60.0 * np.sin(4.0 * np.pi * Y / 3.0) * np.exp(-((X - 5.0) ** 2) / 4.0)
    
    # Add gentle fractal-like micro-roughness
    np.random.seed(42)
    roughness = np.random.normal(0, 1.5, (height, width))
    
    elevation = regional_trend + ridge_1 + ridge_2 - canyon_depth - gullies + roughness
    elevation = np.maximum(elevation, 150.0) # Ensure positive elevation above sea level
    
    # Georeferencing: UTM Zone 30N (WGS84) in meters (e.g. Origin in mountainous catchment)
    # Origin coordinates: X = 600,000 m E, Y = 650,000 m N
    west_origin = 600000.0
    north_origin = 650000.0
    transform = from_origin(west_origin, north_origin, resolution, resolution)
    
    # Write GeoTIFF
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': rasterio.float32,
        'crs': 'EPSG:32630',
        'transform': transform,
        'nodata': -9999.0
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(elevation.astype(np.float32), 1)
        
    print(f"✅ Generated DEM: {output_path} ({width}x{height} at {resolution}m resolution)")
    return output_path, transform, (west_origin, north_origin, resolution)


def generate_road_network(output_path: Path, dem_bounds_info):
    """
    Generates a realistic vector road network crossing the mountain landscape,
    including valley highways and winding mountain traverse roads.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    west_origin, north_origin, resolution = dem_bounds_info
    
    # 4 distinct road corridors crossing ridges and streams:
    # 1. State Highway 101: Low valley arterial running alongside and crossing the river canyon
    line_hwy = LineString([
        (west_origin + 500, north_origin - 9000),
        (west_origin + 2500, north_origin - 7000),
        (west_origin + 5000, north_origin - 5000),
        (west_origin + 7500, north_origin - 3500),
        (west_origin + 9500, north_origin - 1000)
    ])
    
    # 2. Forest Access Road 12: Winding mid-slope timber and logging haul road
    line_forest_rd = LineString([
        (west_origin + 1000, north_origin - 2000),
        (west_origin + 3000, north_origin - 3500),
        (west_origin + 4500, north_origin - 2800),
        (west_origin + 6000, north_origin - 4200),
        (west_origin + 8500, north_origin - 6000)
    ])
    
    # 3. Mountain Ridge Crest Pass: High-elevation steep traverse crossing steep tributary gullies
    line_ridge_pass = LineString([
        (west_origin + 2000, north_origin - 8500),
        (west_origin + 3500, north_origin - 6500),
        (west_origin + 5500, north_origin - 7500),
        (west_origin + 7000, north_origin - 5500),
        (west_origin + 9000, north_origin - 7000)
    ])
    
    # 4. Rural Valley Connector Road
    line_valley_conn = LineString([
        (west_origin + 4000, north_origin - 1000),
        (west_origin + 5200, north_origin - 3500),
        (west_origin + 5800, north_origin - 6500),
        (west_origin + 6500, north_origin - 9500)
    ])
    
    roads_data = [
        {"road_id": "HWY-101", "name": "Mountain Highway 101", "type": "Primary Highway", "speed_kmh": 80, "geometry": line_hwy},
        {"road_id": "FR-12", "name": "Forest Access Road 12", "type": "Secondary Unpaved", "speed_kmh": 40, "geometry": line_forest_rd},
        {"road_id": "PASS-04", "name": "Ridge Crest Pass", "type": "Tertiary Mountain Road", "speed_kmh": 30, "geometry": line_ridge_pass},
        {"road_id": "VAL-02", "name": "Valley Spur Connector", "type": "Rural Collector", "speed_kmh": 50, "geometry": line_valley_conn}
    ]
    
    gdf = gpd.GeoDataFrame(roads_data, crs="EPSG:32630")
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"✅ Generated Road Network: {output_path} ({len(gdf)} road segments)")
    return output_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    dem_path = base_dir / "sample_dem.tif"
    roads_path = base_dir / "sample_roads.geojson"
    
    _, _, bounds_info = generate_mountain_dem(dem_path)
    generate_road_network(roads_path, bounds_info)
