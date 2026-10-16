#!/usr/bin/env python3
"""
Culvert & Slope Hydrological Precondition Engine (v2.0 Advanced)
Executes end-to-end terrain processing, hydrographic routing, road-stream intersection,
Beven-Kirkby Topographic Wetness Index (SMAP 30m downscaling emulation),
Sentinel-1 multi-temporal InSAR ground deformation velocity priors,
USDA Forest Service debris plugging models, and multi-hazard risk ranking.
"""

import argparse
import sys
from pathlib import Path
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np

from src.hydro_routing import (
    fill_depressions,
    calculate_slope_and_aspect,
    calculate_d8_flow_direction,
    calculate_flow_accumulation,
    extract_stream_network
)
from src.road_intersection import (
    detect_culvert_crossings,
    extract_culvert_catchment_attributes
)
from src.soil_moisture_twi import (
    compute_topographic_wetness_index,
    downscale_smap_soil_moisture
)
from src.insar_prior import generate_synthetic_sentinel1_insar_field
from src.risk_engine import assess_culvert_network_risk_advanced
from src.visualizer import build_interactive_risk_map, generate_diagnostic_plots


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Advanced Physics-Informed Culvert Risk & Slope Precondition Engine (v2.0)."
    )
    parser.add_argument(
        "--dem",
        type=str,
        default="data/sample_dem.tif",
        help="Path to Digital Elevation Model GeoTIFF raster."
    )
    parser.add_argument(
        "--roads",
        type=str,
        default="data/sample_roads.geojson",
        help="Path to Road Network vector file (GeoJSON/Shapefile)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output maps, plots, and tabular datasets."
    )
    parser.add_argument(
        "--regional-saturation",
        type=float,
        default=0.70,
        help="Regional antecedent soil moisture saturation ratio m (0.0=dry, 1.0=fully saturated)."
    )
    parser.add_argument(
        "--rainfall",
        type=float,
        default=75.0,
        help="Design storm rainfall intensity in mm/hr (e.g., 25-year design storm)."
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=1.2,
        help="Standard culvert barrel diameter in meters."
    )
    parser.add_argument(
        "--stream-threshold",
        type=float,
        default=0.35,
        help="Minimum upstream contributing drainage area (km^2) to initiate stream channels."
    )
    parser.add_argument(
        "--no-insar",
        action="store_true",
        help="Disable Sentinel-1 InSAR kinematic observational prior coupling."
    )
    parser.add_argument(
        "--no-debris",
        action="store_true",
        help="Disable USDA Forest Service woody debris inlet plugging model."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    base_dir = Path(__file__).resolve().parent
    dem_path = base_dir / args.dem
    roads_path = base_dir / args.roads
    out_dir = base_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 82)
    print("  CULVERT & SLOPE HYDROLOGICAL PRECONDITION ENGINE (v2.0 ADVANCED)")
    print("  Physics-Informed Transportation Infrastructure Resilience & InSAR Coupling")
    print("=" * 82 + "\n")
    
    # 1. Ingest DEM
    if not dem_path.exists():
        print(f"❌ Error: DEM raster not found at {dem_path}")
        print("💡 Tip: Run 'python data/sample_generator.py' to create synthetic test data.")
        sys.exit(1)
        
    print(f"📂 [1/7] Ingesting Elevation Raster: {dem_path.name}")
    with rasterio.open(dem_path) as src:
        dem_raw = src.read(1)
        transform = src.transform
        crs = src.crs.to_string() if src.crs else "EPSG:32630"
        res_x, res_y = src.res
        resolution = float(abs(res_x))
        
    print(f"    Grid Dimensions: {dem_raw.shape[1]}x{dem_raw.shape[0]} cells ({resolution}m pixel resolution)")
    print(f"    Elevation Range: {dem_raw.min():.1f}m to {dem_raw.max():.1f}m (Relief: {dem_raw.max() - dem_raw.min():.1f}m)")
    
    # 2. Hydro-conditioning & Topographic Derivatives
    print("\n⚙️  [2/7] Hydro-Conditioning DEM & Computing Terrain Derivatives...")
    dem = fill_depressions(dem_raw)
    slope_deg, slope_rad = calculate_slope_and_aspect(dem, resolution)
    flow_dir = calculate_d8_flow_direction(dem, resolution)
    accum_cells, accum_km2 = calculate_flow_accumulation(dem, flow_dir, resolution)
    print(f"    Mean Terrain Slope: {slope_deg.mean():.1f}° (Max: {slope_deg.max():.1f}°)")
    print(f"    Max Catchment Accumulation: {accum_km2.max():.2f} km²")
    
    # 3. Topographic Wetness Index & SMAP-HydroBlocks Downscaling Emulation
    print("\n💧 [3/7] Modeling 2D Spatially Distributed Soil Moisture (SMAP-HydroBlocks TWI Emulation)...")
    twi = compute_topographic_wetness_index(accum_cells, slope_rad, resolution)
    saturation_raster_m = downscale_smap_soil_moisture(
        twi, regional_moisture_fraction=args.regional_saturation
    )
    print(f"    Mean Soil Saturation (m): {saturation_raster_m.mean() * 100:.1f}% "
          f"(Hollows: {saturation_raster_m.max() * 100:.0f}%, Ridges: {saturation_raster_m.min() * 100:.0f}%)")
          
    # 4. Sentinel-1 Multi-Temporal InSAR Ground Deformation Simulation
    print("\n📡 [4/7] Generating Sentinel-1 Multi-Temporal InSAR LOS Deformation Velocity Field...")
    insar_velocity = generate_synthetic_sentinel1_insar_field(slope_deg, saturation_raster_m)
    active_creep_cells = np.sum(insar_velocity <= -10.0)
    print(f"    InSAR LOS Velocity Range: {insar_velocity.min():.1f} to {insar_velocity.max():.1f} mm/year")
    print(f"    Active Downslope Creep Cells (v_LOS ≤ -10 mm/yr): {active_creep_cells} pixels")
    
    # 5. Stream Network & Road-Stream Intersections
    print(f"\n🌊 [5/7] Extracting Stream Channels (≥{args.stream_threshold} km²) & Detecting Road Crossings...")
    streams_gdf = extract_stream_network(accum_km2, transform, crs, threshold_km2=args.stream_threshold)
    roads_gdf = gpd.read_file(roads_path)
    crossings_gdf = detect_culvert_crossings(roads_gdf, streams_gdf)
    print(f"    Identified {len(crossings_gdf)} physical road-stream crossing intersections.")
    
    if len(crossings_gdf) == 0:
        print("⚠️ Warning: No road-stream intersections detected.")
        sys.exit(0)
        
    print("    Delineating upstream catchments and slope statistics for each culvert...")
    attributed_culverts = extract_culvert_catchment_attributes(
        crossings_gdf, dem, slope_deg, flow_dir, accum_km2, transform, resolution
    )
    
    # 6. Advanced Physics & Multi-Hazard Assessment
    print(f"\n🔬 [6/7] Running Advanced Multi-Hazard Engine (InSAR Prior + Debris Plugging + TWI)...")
    assessed_culverts = assess_culvert_network_risk_advanced(
        culverts_gdf=attributed_culverts,
        saturation_raster_m=saturation_raster_m,
        insar_velocity_raster=insar_velocity,
        dem_transform=transform,
        regional_saturation_scalar=args.regional_saturation,
        design_rainfall_intensity_mm_hr=args.rainfall,
        default_pipe_diameter_m=args.diameter,
        enable_insar_coupling=not args.no_insar,
        enable_debris_plugging=not args.no_debris
    )
    
    # 7. Export Results, Web GIS Map & Diagnostic Charts
    print("\n📊 [7/7] Generating Executive Outputs, Web GIS Map & Diagnostic Charts...")
    csv_path = out_dir / "culvert_risk_assessment_summary.csv"
    geojson_path = out_dir / "culvert_crossings_assessed.geojson"
    
    export_df = pd.DataFrame(assessed_culverts.drop(columns="geometry"))
    export_df.to_csv(csv_path, index=False)
    assessed_culverts.to_file(geojson_path, driver="GeoJSON")
    print(f"    Saved Tabular Report: {csv_path.name}")
    print(f"    Saved GeoJSON: {geojson_path.name}")
    
    html_map_path = out_dir / "culvert_risk_map.html"
    build_interactive_risk_map(assessed_culverts, roads_gdf, streams_gdf, html_map_path)
    
    plot_path = out_dir / "culvert_risk_diagnostics.png"
    generate_diagnostic_plots(assessed_culverts, plot_path)
    
    # Console Summary Table
    print("\n" + "=" * 105)
    print("  EXECUTIVE CULVERT VULNERABILITY SUMMARY (RANKED BY RISK - v2.0 ADVANCED)")
    print("=" * 105)
    print(f"{'ID':<10} {'Road Name':<22} {'Area(km²)':<10} {'Slope':<7} {'FS_prior':<9} {'InSAR(mm/y)':<12} {'FS_insar':<9} {'HR':<6} {'CVI':<6} {'Risk Tier'}")
    print("-" * 105)
    
    for _, row in assessed_culverts.iterrows():
        print(f"{row['culvert_id']:<10} "
              f"{str(row['road_name'])[:20]:<22} "
              f"{row['drainage_area_km2']:<10.2f} "
              f"{row['p90_upslope_deg']:<6.1f}° "
              f"{row['factor_of_safety_prior']:<9.2f} "
              f"{row['insar_velocity_los_mmyr']:<12.1f} "
              f"{row['factor_of_safety_insar']:<9.2f} "
              f"{row['hydraulic_surcharge_ratio']:<6.2f} "
              f"{row['composite_vulnerability_index']:<6.1f} "
              f"{row['risk_tier']}")
              
    print("=" * 105)
    print(f"\n🎉 Advanced v2.0 Pipeline executed successfully! Interactive map available at:\n   {html_map_path}\n")


if __name__ == "__main__":
    main()
