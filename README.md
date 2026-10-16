# 🌊 Culvert & Slope Hydrological Precondition Engine
### Physics-Informed Computational Hydrology & Geotechnical Infrastructure Risk Framework

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-11%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain](https://img.shields.io/badge/Domain-Hydrology%20%7C%20Geotechnics%20%7C%20InSAR-purple.svg)]()

> **Author:** Peter Aryee  
> **Contact:** [paryee.geospatial@gmail.com](mailto:paryee.geospatial@gmail.com) | [LinkedIn](https://linkedin.com/in/peter-aryee-geospatial)  
> **Academic Focus:** Department of Earth, Environmental and Planetary Sciences (EEPS), Rice University  
> **Target Initiative:** USDA Forest Service CulvertApp (`culvert-at-risk.org`) & Physics-Informed AI for Landslides

---

## 1. Scientific Motivation & Real-World Problem

Across the United States and global mountainous terrains, transportation corridors intersect natural drainage networks at millions of **road-stream culvert crossings**. During extreme storm events, culverts frequently suffer catastrophic structural blowouts through a dual-mechanism coupling:

```
  [ Extreme Rainfall / Antecedent Wetness ]
                      │
                      ▼
        [ Upslope Soil Saturation ] ──► [ Elevated Pore Pressure: u = m * γ_w * z * cos²(β) ]
                      │                                      │
                      ▼                                      ▼
        [ Runoff Surge: Q_peak ]                  [ Geotechnical Slope Failure: FS < 1.0 ]
                      │                                      │
                      ▼                                      ▼
   ═══════════════════[ ROADWAY EMBANKMENT ]═════════════════════════════════
                      [ CULVERT BARREL ] ◄── Sediment Infill & Hydraulic Surcharge
   ──────────────────────────────────────────────────────────────────────────
                      │
                      ▼
         💥 [ CATASTROPHIC EMBANKMENT WASHOUT / ROAD BLOWOUT ]
```

1. **Hydraulic Surcharge:** Storm peak discharge $Q_{\text{peak}}$ exceeds the culvert pipe conveyance capacity $Q_{\text{cap}}$, causing backwater to pond against the unengineered road embankment.
2. **Geotechnical Instability:** Pre-existing antecedent soil moisture ($m = z_w / z$) elevates pore-water pressure ($u$), reducing effective normal stress ($\sigma' = \sigma - u$) and triggering shallow translational landslides and debris flows that choke the culvert inlet.

This open-source engine automates the end-to-end detection, catchment delineation, geotechnical stability modeling, and multi-hazard risk ranking of culverts using standard Digital Elevation Models (DEMs) and vector road networks.

---

## 2. Mathematical & Physical Formulation

### A. Geotechnical Slope Stability (Terzaghi-Coulomb Infinite Slope Model)
The Factor of Safety ($FS$) against translational shear failure along an inclined planar failure surface is expressed as the ratio of resisting shear strength to gravitational driving shear stress:

$$FS = \frac{\tau_f}{\tau_d} = \frac{c' + (\sigma - u)\tan\phi'}{\gamma_s z \sin\beta \cos\beta}$$

Incorporating antecedent hydrologic saturation ratio $m = \frac{z_w}{z}$ (where $z_w$ is the height of the perched water table above the slip surface, $0 \le m \le 1$):

$$FS = \frac{c' + (\gamma_s - m \gamma_w) z \cos^2\beta \tan\phi'}{\gamma_s z \sin\beta \cos\beta}$$

* $c'$: Effective soil cohesion ($\text{kPa}$)
* $\phi'$: Internal soil friction angle ($\text{degrees}$)
* $\beta$: Topographic hillslope gradient ($\text{degrees}$, derived from DEM)
* $z$: Soil mantle depth down to bedrock slip surface ($\text{m}$)
* $\gamma_s, \gamma_w$: Unit weights of bulk soil ($18.5 \text{ kN/m}^3$) and water ($9.81 \text{ kN/m}^3$)
* $m$: **Hydrological Saturation Ratio** ($0.0 = \text{dry}$, $1.0 = \text{fully saturated}$)

**Stability Thresholds:**
* $FS \ge 1.5$: Geotechnically Stable
* $1.2 \le FS < 1.5$: Marginally Stable (vulnerable to high antecedent rain)
* $1.0 \le FS < 1.2$: High Hazard (near failure boundary)
* $FS < 1.0$: **Active Slope Failure / Liquefaction Trigger**

---

### B. Analytical Critical Saturation Threshold ($m_{\text{crit}}$)
By setting $FS = 1.0$, the engine analytically computes the exact soil moisture threshold $m_{\text{crit}}$ required to trigger failure:

$$m_{\text{crit}} = \frac{c' + \gamma_s z \cos^2\beta \tan\phi' - \gamma_s z \sin\beta \cos\beta}{\gamma_w z \cos^2\beta \tan\phi'}$$

* If $m_{\text{crit}} > 1.0$: Slope remains stable even under 100% saturation.
* If $m_{\text{crit}} \le 0.0$: Slope is unconditionally unstable even when dry.

---

### C. Hydraulic Peak Runoff & Pipe Conveyance Capacity
Peak storm discharge is evaluated using the Rational / TR-55 hydrographic formulation:

$$Q_{\text{peak}} = \frac{C \cdot I \cdot A_{\text{catchment}}}{3.6} \quad [\text{m}^3/\text{s}]$$

Culvert full-pipe hydraulic conveyance capacity is governed by Manning's gravity flow equation:

$$Q_{\text{cap}} = \frac{1}{n} A_{\text{pipe}} R_h^{2/3} S_0^{1/2} \quad [\text{m}^3/\text{s}]$$

Where $A_{\text{pipe}} = \frac{\pi D^2}{4}$, hydraulic radius $R_h = \frac{D}{4}$, $S_0$ is culvert barrel slope, and $n$ is Manning's roughness coefficient (0.024 for corrugated metal pipe).

---

### D. Composite Vulnerability Index (CVI) & Blowout Multiplier
Culverts are evaluated through a multi-hazard index combining geotechnical slope vulnerability, hydraulic surcharge ratio $HR = Q_{\text{peak}} / Q_{\text{cap}}$, and roadway consequence weight $W_{\text{road}}$:

$$\text{CVI} = \Big[ 0.50 \cdot S_{\text{geotech}} + 0.50 \cdot S_{\text{hydraulic}} \Big] \cdot W_{\text{road}} \cdot \Psi_{\text{blowout}}$$

Where $\Psi_{\text{blowout}} = 1.35$ is the **Embankment Blowout Hazard Multiplier**, triggered whenever severe hydraulic overtopping ($HR \ge 1.0$) directly coincides with upslope geotechnical instability ($FS \le 1.25$).

---

## 3. Architecture & Modular Design

```
culvert_slope_mini_project/
├── data/
│   ├── sample_generator.py      # Synthetic high-relief mountain terrain & road vector generator
│   ├── sample_dem.tif           # 25m DEM GeoTIFF (EPSG:32630)
│   └── sample_roads.geojson     # Road network vector corridors
├── src/
│   ├── hydro_routing.py         # DEM conditioning, finite-difference slope, D8 flow accumulation
│   ├── road_intersection.py     # Road-stream spatial overlay & upstream catchment delineation
│   ├── slope_physics.py         # Terzaghi-Coulomb FS modeling & m_crit analytical solver
│   ├── hydraulic_capacity.py    # Peak storm runoff & Manning culvert flow capacity
│   ├── risk_engine.py           # Multi-hazard risk prioritization & blowout scoring
│   └── visualizer.py            # Interactive Folium Web GIS map & 3-panel diagnostic figures
├── notebooks/
│   └── 01_pipeline_walkthrough.ipynb  # End-to-end interactive Jupyter demonstration
├── tests/
│   ├── test_hydro_routing.py    # D8 flow direction, gradient & accumulation tests
│   ├── test_slope_physics.py    # Analytical FS and moisture sensitivity tests
│   └── test_hydraulic_capacity.py # Manning's conveyance & discharge scaling tests
├── output/
│   ├── culvert_risk_map.html    # Interactive Web GIS Folium map
│   ├── culvert_risk_diagnostics.png # High-resolution 300 DPI publication plots
│   └── culvert_risk_assessment_summary.csv # Tabular engineering dataset
├── run_pipeline.py              # CLI driver with configurable parameters
├── requirements.txt             # Lightweight Python dependencies
└── pyproject.toml               # Python packaging metadata
```

---

## 4. Quickstart Guide

### Prerequisites
* Python 3.9+
* Required libraries: `rasterio`, `geopandas`, `shapely`, `numpy`, `scipy`, `matplotlib`, `folium`, `pandas`, `pytest`

### Installation
```bash
git clone https://github.com/eminent337/culvert-slope-risk-engine.git
cd culvert-slope-risk-engine
pip install -r requirements.txt
```

### Run Full Analysis Pipeline
```bash
python run_pipeline.py --saturation 0.70 --rainfall 75.0 --diameter 1.2
```

### Run Unit Tests
```bash
pytest tests/ -v
```

---

## 5. Visual Outputs & Cartographic Products

### Interactive Web GIS Dashboard (`output/culvert_risk_map.html`)
The pipeline automatically generates a Leaflet/Folium web map featuring:
* Color-coded crossing pins: **Green** (Low Risk) $\rightarrow$ **Yellow** (Moderate) $\rightarrow$ **Orange** (High) $\rightarrow$ **Red** (Critical Blowout Hazard).
* Interactive engineering popups displaying:
  * Culvert ID & Road Name
  * Composite Vulnerability Index (CVI)
  * Factor of Safety ($FS$) & Critical Saturation ($m_{\text{crit}}$)
  * Peak Inflow ($Q_{\text{peak}}$) vs Pipe Conveyance ($Q_{\text{cap}}$)
  * Upstream Catchment Area ($\text{km}^2$) and 90th percentile slope gradient.

### Diagnostic Scientific Plots (`output/culvert_risk_diagnostics.png`)
* **Panel A:** Factor of Safety degradation curves across the saturation continuum ($m = 0.0 \rightarrow 1.0$) identifying critical slope tipping points.
* **Panel B:** Peak storm discharge vs culvert barrel capacity under design rainfall.
* **Panel C:** Culvert network risk prioritization ranking.

---

## 6. Scientific Foundations & Prior Art Citations

This framework synthesizes established methodologies from computational hydrology and geotechnical infrastructure analysis:

1. **USDA Forest Service GRAIP (Geomorphic Road Analysis and Inventory Package):**
   * Cissel, J. M., et al. (2014). *GRAIP-Lite: A tool for evaluating road sediment delivery and culvert failure risk across watershed scales*. USDA Forest Service Rocky Mountain Research Station.
2. **Cornell Culvert Evaluation Model:**
   * Soil and Water Lab, Cornell University (2018). *Assessing flood risk and hydraulic capacity for road-stream crossings using TR-55 runoff methodology*.
3. **Landlab Earth-Surface Dynamics Framework:**
   * Hobley, D. E. J., et al. (2017). *Creative computing with Landlab: an open-source library for building 2D numerical models of Earth-surface dynamics*. Earth Surface Dynamics.
4. **Hyper-Resolution Hydrology & Antecedent Moisture Preconditions:**
   * Vergopolan, N., et al. (2021). *SMAP-HydroBlocks: A 30-m satellite-based soil moisture dataset for hyper-resolution environmental modeling across the United States*. Scientific Data.

---

## 7. License

Distributed under the **MIT License**. See `LICENSE` for details.
