"""
Geospatial Visualization & Cartographic Reporting Engine.
Builds interactive web GIS maps (Folium) and publication-quality diagnostic plots (Matplotlib).
"""

import folium
from folium import plugins
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Optional


def build_interactive_risk_map(
    culverts_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    streams_gdf: gpd.GeoDataFrame,
    output_html_path: Path
) -> Path:
    """
    Renders a full-featured interactive web GIS map with multi-layer controls,
    color-coded culvert risk nodes, and diagnostic popup cards.
    """
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Reproject all spatial layers to WGS84 (EPSG:4326) for Leaflet/Folium mapping
    culv_wgs84 = culverts_gdf.to_crs(epsg=4326)
    roads_wgs84 = roads_gdf.to_crs(epsg=4326)
    streams_wgs84 = streams_gdf.to_crs(epsg=4326)
    
    # Calculate map center from culvert points
    center_lat = culv_wgs84.geometry.y.mean()
    center_lon = culv_wgs84.geometry.x.mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True
    )
    
    # Alternative Base Tiles
    folium.TileLayer("OpenStreetMap", name="Street View").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark Matter (High Contrast)").add_to(m)
    
    # 1. Hydrographic Stream Network Layer
    streams_group = folium.FeatureGroup(name="Hydrological Stream Network (D8)", show=True)
    for _, stream in streams_wgs84.iterrows():
        coords = [(y, x) for x, y in stream.geometry.coords]
        folium.PolyLine(
            locations=coords,
            color="#2980B9",
            weight=3.0,
            opacity=0.85,
            tooltip=f"Stream Reach: {stream.get('reach_id', 'Reach')}"
        ).add_to(streams_group)
    streams_group.add_to(m)
    
    # 2. Road Infrastructure Network Layer
    roads_group = folium.FeatureGroup(name="Transportation Network", show=True)
    for _, road in roads_wgs84.iterrows():
        coords = [(y, x) for x, y in road.geometry.coords]
        rtype = road.get("type", "Road")
        color = "#E67E22" if "Highway" in rtype else "#7F8C8D"
        weight = 4.5 if "Highway" in rtype else 2.5
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=weight,
            opacity=0.9,
            tooltip=f"{road.get('name', 'Road')} ({rtype})"
        ).add_to(roads_group)
    roads_group.add_to(m)
    
    # 3. Culvert Crossings & Vulnerability Assessment Layer
    culverts_group = folium.FeatureGroup(name="Culvert Crossings (Multi-Hazard Risk)", show=True)
    
    for _, c in culv_wgs84.iterrows():
        lat = c.geometry.y
        lon = c.geometry.x
        cid = c.get("culvert_id", "CULVERT")
        road_name = c.get("road_name", "Road")
        cvi = c.get("composite_vulnerability_index", 0.0)
        risk_tier = c.get("risk_tier", "LOW")
        color = c.get("risk_color", "#2ECC71")
        fs = c.get("factor_of_safety", 1.5)
        m_crit = c.get("critical_saturation_mcrit", 1.0)
        q_peak = c.get("q_peak_m3s", 0.0)
        q_cap = c.get("q_capacity_m3s", 0.0)
        hr = c.get("hydraulic_surcharge_ratio", 0.0)
        slope = c.get("p90_upslope_deg", 15.0)
        area = c.get("drainage_area_km2", 0.5)
        blowout = c.get("blowout_hazard", False)
        
        blowout_badge = '<span style="background-color:#E74C3C;color:white;padding:2px 6px;border-radius:4px;font-weight:bold;">⚠️ BLOWOUT HAZARD</span>' if blowout else '<span style="background-color:#2ECC71;color:white;padding:2px 6px;border-radius:4px;">Embankment Intact</span>'
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 260px; font-size: 13px;">
            <div style="background-color: {color}; color: white; padding: 8px; border-radius: 4px 4px 0 0; font-weight: bold;">
                <span style="font-size: 15px;">{cid}</span> — {risk_tier}
            </div>
            <div style="padding: 10px; border: 1px solid #ddd; border-top: none; background-color: #fafafa;">
                <p style="margin: 4px 0;"><strong>Roadway:</strong> {road_name}</p>
                <p style="margin: 4px 0;"><strong>Vulnerability Index (CVI):</strong> <span style="font-size: 15px; font-weight: bold; color: {color};">{cvi} / 100</span></p>
                <div style="margin: 6px 0;">{blowout_badge}</div>
                <hr style="margin: 8px 0; border: 0; border-top: 1px solid #ccc;">
                <p style="margin: 3px 0;"><strong>Geotechnical Stability (FS):</strong> <b>{fs:.2f}</b></p>
                <p style="margin: 3px 0;"><strong>Critical Saturation (m_crit):</strong> <b>{m_crit:.2f}</b></p>
                <p style="margin: 3px 0;"><strong>Peak Inflow (Q_peak):</strong> <b>{q_peak:.2f} m³/s</b></p>
                <p style="margin: 3px 0;"><strong>Pipe Flow Capacity (Q_cap):</strong> <b>{q_cap:.2f} m³/s</b></p>
                <p style="margin: 3px 0;"><strong>Hydraulic Surcharge (HR):</strong> <b>{hr:.2f}x</b></p>
                <p style="margin: 3px 0;"><strong>Upstream Catchment:</strong> <b>{area:.2f} km²</b> | <b>{slope:.1f}°</b> slope</p>
            </div>
        </div>
        """
        
        # Circle Marker
        radius = 8.0 if cvi < 50 else 12.0
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color="#2C3E50",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"<b>{cid}</b>: {risk_tier} (CVI: {cvi})"
        ).add_to(culverts_group)
        
    culverts_group.add_to(m)
    
    # 4. Floating Legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 30px; right: 30px; width: 220px; z-index:9999; font-size:12px;
                background-color: white; padding: 12px; border-radius: 6px; 
                border: 2px solid #7f8c8d; box-shadow: 2px 2px 8px rgba(0,0,0,0.3);">
        <b style="font-size: 13px;">Culvert Risk Classification</b><br>
        <span style="font-size: 11px; color: #7f8c8d;">Antecedent Saturation & Peak Flow</span>
        <div style="margin-top: 8px;">
            <i style="background: #E74C3C; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i><b>CRITICAL (Blowout)</b><br>
            <i style="background: #E67E22; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i><b>HIGH RISK</b><br>
            <i style="background: #F1C40F; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i><b>MODERATE RISK</b><br>
            <i style="background: #2ECC71; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i><b>LOW RISK</b>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Layer switcher
    folium.LayerControl(collapsed=False).add_to(m)
    
    m.save(str(output_html_path))
    print(f"✅ Generated Interactive Map: {output_html_path}")
    return output_html_path


def generate_diagnostic_plots(
    culverts_gdf: gpd.GeoDataFrame,
    output_png_path: Path
) -> Path:
    """
    Renders publication-grade 3-panel scientific diagnostic chart:
    Panel A: Factor of Safety degradation curves under soil saturation
    Panel B: Hydraulic Load vs Capacity
    Panel C: Composite Vulnerability Index ranking
    """
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    plt.subplots_adjust(wspace=0.3)
    
    # Panel A: Factor of Safety vs Saturation m
    m_range = np.linspace(0.0, 1.0, 50)
    ax1 = axes[0]
    
    top_culverts = culverts_gdf.head(5)
    for _, c in top_culverts.iterrows():
        slope = float(c.get("p90_upslope_deg", 15.0))
        cid = c["culvert_id"]
        from .slope_physics import calculate_factor_of_safety
        fs_curve = [calculate_factor_of_safety(slope, saturation_ratio_m=m) for m in m_range]
        ax1.plot(m_range, fs_curve, label=f"{cid} ({slope:.0f}° slope)", lw=2.2)
        
    ax1.axhline(1.0, color='red', linestyle='--', lw=2, label='Failure Threshold (FS = 1.0)')
    ax1.axhline(1.5, color='green', linestyle=':', lw=1.5, label='Design Stability (FS = 1.5)')
    ax1.set_title("A. Slope Factor of Safety vs Soil Saturation", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Antecedent Saturation Ratio (m = zw / z)", fontsize=11)
    ax1.set_ylabel("Geotechnical Factor of Safety (FS)", fontsize=11)
    ax1.set_ylim(0.2, 3.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # Panel B: Peak Storm Flow vs Culvert Pipe Capacity
    ax2 = axes[1]
    ids = culverts_gdf["culvert_id"].tolist()
    q_peaks = culverts_gdf["q_peak_m3s"].tolist()
    q_caps = culverts_gdf["q_capacity_m3s"].tolist()
    
    x = np.arange(len(ids))
    width = 0.35
    
    ax2.bar(x - width/2, q_peaks, width, label='Peak Runoff (Q_peak)', color='#E74C3C', alpha=0.85)
    ax2.bar(x + width/2, q_caps, width, label='Pipe Capacity (Q_cap)', color='#3498DB', alpha=0.85)
    
    ax2.set_title("B. Storm Runoff vs Culvert Pipe Conveyance", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Culvert Identifier", fontsize=11)
    ax2.set_ylabel("Discharge Rate (m³/s)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ids, rotation=35, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper left', fontsize=9)
    
    # Panel C: Composite Vulnerability Index (CVI) Ranking
    ax3 = axes[2]
    cvis = culverts_gdf["composite_vulnerability_index"].tolist()
    colors = culverts_gdf["risk_color"].tolist()
    
    bars = ax3.barh(ids[::-1], cvis[::-1], color=colors[::-1], edgecolor='black', alpha=0.85)
    ax3.set_title("C. Composite Vulnerability Index (CVI)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Vulnerability Score (0 - 100)", fontsize=11)
    ax3.set_xlim(0, 105)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add values on bar ends
    for bar in bars:
        w = bar.get_width()
        ax3.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}", va='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    fig.savefig(output_png_path, dpi=300)
    plt.close(fig)
    print(f"✅ Generated Diagnostic Plots: {output_png_path}")
    return output_png_path
