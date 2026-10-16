"""
Advanced Geospatial Visualization & Diagnostic Cartographic Engine (v2.0).
Renders interactive multi-layer web GIS maps (Folium) and 4-panel publication figures (Matplotlib)
incorporating InSAR ground deformation kinematics and USDA debris plugging diagnostics.
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
    Renders an advanced interactive web GIS map with multi-hazard engineering popups,
    InSAR kinematic badges, and layer controls.
    """
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Reproject all spatial layers to WGS84 (EPSG:4326) for Leaflet/Folium mapping
    culv_wgs84 = culverts_gdf.to_crs(epsg=4326)
    roads_wgs84 = roads_gdf.to_crs(epsg=4326)
    streams_wgs84 = streams_gdf.to_crs(epsg=4326)
    
    center_lat = culv_wgs84.geometry.y.mean()
    center_lon = culv_wgs84.geometry.x.mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True
    )
    
    # Base Tiles
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
    
    # 3. Culvert Crossings Layer
    culverts_group = folium.FeatureGroup(name="Culvert Crossings (Multi-Hazard Risk)", show=True)
    
    for _, c in culv_wgs84.iterrows():
        lat = c.geometry.y
        lon = c.geometry.x
        cid = c.get("culvert_id", "CULVERT")
        road_name = c.get("road_name", "Road")
        cvi = c.get("composite_vulnerability_index", 0.0)
        risk_tier = c.get("risk_tier", "LOW")
        color = c.get("risk_color", "#2ECC71")
        
        fs_prior = c.get("factor_of_safety_prior", c.get("factor_of_safety", 1.5))
        fs_insar = c.get("factor_of_safety_insar", fs_prior)
        v_los = c.get("insar_velocity_los_mmyr", 0.0)
        insar_status = c.get("insar_status", "Stationary")
        
        q_peak = c.get("q_peak_m3s", 0.0)
        q_eff = c.get("q_effective_capacity_m3s", c.get("q_capacity_m3s", 0.0))
        hr = c.get("hydraulic_surcharge_ratio", 0.0)
        p_debris = c.get("debris_plugging_prob", 0.0)
        debris_tier = c.get("debris_risk_tier", "Low")
        
        slope = c.get("p90_upslope_deg", 15.0)
        area = c.get("drainage_area_km2", 0.5)
        m_sat = c.get("local_saturation_m", 0.70)
        blowout = c.get("blowout_hazard", False)
        
        blowout_badge = '<span style="background-color:#E74C3C;color:white;padding:2px 6px;border-radius:4px;font-weight:bold;">⚠️ BLOWOUT HAZARD</span>' if blowout else '<span style="background-color:#2ECC71;color:white;padding:2px 6px;border-radius:4px;">Embankment Intact</span>'
        
        insar_badge = f'<span style="background-color:#8E44AD;color:white;padding:2px 6px;border-radius:4px;font-size:11px;">📡 InSAR: {v_los:.1f} mm/yr ({insar_status})</span>'
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 280px; font-size: 13px;">
            <div style="background-color: {color}; color: white; padding: 8px; border-radius: 4px 4px 0 0; font-weight: bold;">
                <span style="font-size: 15px;">{cid}</span> — {risk_tier}
            </div>
            <div style="padding: 10px; border: 1px solid #ddd; border-top: none; background-color: #fafafa;">
                <p style="margin: 3px 0;"><strong>Roadway:</strong> {road_name}</p>
                <p style="margin: 3px 0;"><strong>Vulnerability Index (CVI):</strong> <span style="font-size: 15px; font-weight: bold; color: {color};">{cvi} / 100</span></p>
                <div style="margin: 5px 0;">{blowout_badge}</div>
                <div style="margin: 5px 0;">{insar_badge}</div>
                <hr style="margin: 8px 0; border: 0; border-top: 1px solid #ccc;">
                <p style="margin: 3px 0;"><strong>Static Geotech Stability (FS_prior):</strong> <b>{fs_prior:.2f}</b></p>
                <p style="margin: 3px 0;"><strong>InSAR-Updated Stability (FS_insar):</strong> <b style="color:{color};">{fs_insar:.2f}</b></p>
                <p style="margin: 3px 0;"><strong>Local Moisture Saturation (m):</strong> <b>{m_sat * 100:.0f}% (TWI)</b></p>
                <p style="margin: 3px 0;"><strong>Woody Debris Jamming Risk:</strong> <b>{p_debris * 100:.0f}% ({debris_tier})</b></p>
                <p style="margin: 3px 0;"><strong>Peak Storm Inflow (Q_peak):</strong> <b>{q_peak:.2f} m³/s</b></p>
                <p style="margin: 3px 0;"><strong>Effective Conveyance (Q_eff):</strong> <b>{q_eff:.2f} m³/s</b></p>
                <p style="margin: 3px 0;"><strong>Hydraulic Surcharge Ratio (HR):</strong> <b>{hr:.2f}x</b></p>
                <p style="margin: 3px 0;"><strong>Contributing Drainage Area:</strong> <b>{area:.2f} km²</b> (Slope: <b>{slope:.1f}°</b>)</p>
            </div>
        </div>
        """
        
        radius = 8.0 if cvi < 40 else (11.0 if cvi < 70 else 14.0)
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color="#2C3E50",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"<b>{cid}</b>: {risk_tier} (CVI: {cvi} | InSAR: {v_los:.1f} mm/yr)"
        ).add_to(culverts_group)
        
    culverts_group.add_to(m)
    
    # Legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 30px; right: 30px; width: 240px; z-index:9999; font-size:12px;
                background-color: white; padding: 12px; border-radius: 6px; 
                border: 2px solid #7f8c8d; box-shadow: 2px 2px 8px rgba(0,0,0,0.3);">
        <b style="font-size: 13px;">Advanced Culvert Risk (v2.0)</b><br>
        <span style="font-size: 11px; color: #7f8c8d;">InSAR Creep + TWI Saturation + Debris</span>
        <div style="margin-top: 8px;">
            <i style="background: #E74C3C; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i><b>CRITICAL (Blowout)</b><br>
            <i style="background: #E67E22; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i><b>HIGH RISK</b><br>
            <i style="background: #F1C40F; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i><b>MODERATE RISK</b><br>
            <i style="background: #2ECC71; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i><b>LOW RISK</b>
        </div>
        <hr style="margin: 6px 0;">
        <span style="font-size: 11px; color: #555;">📡 Pulsing: Active InSAR ground creep</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    
    m.save(str(output_html_path))
    print(f"✅ Generated Advanced Interactive Map: {output_html_path}")
    return output_html_path


def generate_diagnostic_plots(
    culverts_gdf: gpd.GeoDataFrame,
    output_png_path: Path
) -> Path:
    """
    Renders publication-grade 4-panel (2x2) scientific diagnostic chart:
    Panel A: Factor of Safety degradation curves under soil saturation
    Panel B: Sentinel-1 InSAR LOS velocity vs Kinematic FS Degradation
    Panel C: Peak Storm Runoff vs Nominal & Debris-Constricted Capacity
    Panel D: Composite Vulnerability Index ranking
    """
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.25)
    
    # Panel A: Factor of Safety vs Saturation m
    ax1 = axes[0, 0]
    m_range = np.linspace(0.0, 1.0, 50)
    top_culverts = culverts_gdf.head(5)
    for _, c in top_culverts.iterrows():
        slope = float(c.get("p90_upslope_deg", 15.0))
        cid = c["culvert_id"]
        from .slope_physics import calculate_factor_of_safety
        fs_curve = [calculate_factor_of_safety(slope, saturation_ratio_m=m) for m in m_range]
        ax1.plot(m_range, fs_curve, label=f"{cid} ({slope:.0f}° slope)", lw=2.2)
        
    ax1.axhline(1.0, color='red', linestyle='--', lw=2, label='Failure Threshold (FS = 1.0)')
    ax1.axhline(1.5, color='green', linestyle=':', lw=1.5, label='Design Target (FS = 1.5)')
    ax1.set_title("A. Slope Factor of Safety vs Antecedent Moisture", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Antecedent Saturation Ratio (m = zw / z)", fontsize=10)
    ax1.set_ylabel("Factor of Safety (FS)", fontsize=10)
    ax1.set_ylim(0.2, 3.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8.5)
    
    # Panel B: InSAR Velocity vs Kinematic FS Degradation
    ax2 = axes[0, 1]
    v_los_vals = culverts_gdf["insar_velocity_los_mmyr"].tolist()
    fs_priors = culverts_gdf["factor_of_safety_prior"].tolist()
    fs_insars = culverts_gdf["factor_of_safety_insar"].tolist()
    ids = culverts_gdf["culvert_id"].tolist()
    
    scatter = ax2.scatter(v_los_vals, fs_insars, c=culverts_gdf["composite_vulnerability_index"], 
                          cmap='RdYlGn_r', s=120, edgecolors='black', zorder=4)
    for i, cid in enumerate(ids):
        ax2.annotate(cid, (v_los_vals[i], fs_insars[i]), textcoords="offset points", 
                     xytext=(6, 6), fontsize=8, fontweight='bold')
                     
    ax2.axvline(-12.0, color='purple', linestyle='--', lw=1.8, label='Active InSAR Creep Threshold (-12 mm/yr)')
    ax2.axhline(1.0, color='red', linestyle='--', lw=2, label='Geotechnical Failure (FS = 1.0)')
    ax2.set_title("B. Sentinel-1 InSAR Velocity vs Updated Stability", fontsize=12, fontweight='bold')
    ax2.set_xlabel("InSAR LOS Velocity (mm/year; negative = downslope creep)", fontsize=10)
    ax2.set_ylabel("InSAR-Updated Factor of Safety (FS_insar)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Vulnerability Index (CVI)')
    ax2.legend(loc='upper left', fontsize=8.5)
    
    # Panel C: Peak Storm Flow vs Pipe Capacity (Nominal vs Debris-Reduced)
    ax3 = axes[1, 0]
    q_peaks = culverts_gdf["q_peak_m3s"].tolist()
    q_caps_nom = culverts_gdf["q_nominal_capacity_m3s"].tolist()
    q_caps_eff = culverts_gdf["q_effective_capacity_m3s"].tolist()
    
    x = np.arange(len(ids))
    width = 0.28
    
    ax3.bar(x - width, q_peaks, width, label='Storm Peak Runoff (Q_peak)', color='#E74C3C', alpha=0.9)
    ax3.bar(x, q_caps_nom, width, label='Nominal Pipe Capacity', color='#3498DB', alpha=0.8)
    ax3.bar(x + width, q_caps_eff, width, label='Effective Capacity (Debris Reduced)', color='#E67E22', alpha=0.9)
    
    ax3.set_title("C. Hydraulic Runoff vs Debris-Constricted Conveyance", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Culvert Crossing Identifier", fontsize=10)
    ax3.set_ylabel("Discharge (m³/s)", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(ids, rotation=35, ha='right', fontsize=8.5)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='upper left', fontsize=8.5)
    
    # Panel D: Composite Vulnerability Index Ranking
    ax4 = axes[1, 1]
    cvis = culverts_gdf["composite_vulnerability_index"].tolist()
    colors = culverts_gdf["risk_color"].tolist()
    
    bars = ax4.barh(ids[::-1], cvis[::-1], color=colors[::-1], edgecolor='black', alpha=0.85)
    ax4.set_title("D. Advanced Composite Vulnerability Index (CVI)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Composite Risk Score (0 - 100)", fontsize=10)
    ax4.set_xlim(0, 105)
    ax4.grid(True, alpha=0.3, axis='x')
    
    for bar in bars:
        w = bar.get_width()
        ax4.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{w:.1f}", va='center', fontsize=8.5, fontweight='bold')
        
    plt.tight_layout()
    fig.savefig(output_png_path, dpi=300)
    plt.close(fig)
    print(f"✅ Generated 4-Panel Diagnostic Plots: {output_png_path}")
    return output_png_path
