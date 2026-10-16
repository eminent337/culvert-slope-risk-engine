# 🌊 Culvert & Slope Hydrological Precondition Engine (v2.0)
### Physics-Informed Computational Hydrology, InSAR Deformation Priors & Infrastructure Resilience

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-17%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Radar](https://img.shields.io/badge/Remote%20Sensing-Sentinel--1%20InSAR-purple.svg)]()
[![Hydrology](https://img.shields.io/badge/Hydrology-SMAP--HydroBlocks%20TWI-blue.svg)]()

> **Author:** Peter Aryee  
> **Contact:** [paryee.geospatial@gmail.com](mailto:paryee.geospatial@gmail.com) | [LinkedIn](https://linkedin.com/in/peter-aryee-geospatial)  
> **Academic Target:** Department of Earth, Environmental and Planetary Sciences (EEPS), Rice University  
> **Target Initiative:** USDA Forest Service CulvertApp (`culvert-at-risk.org`) & Physics-Informed AI for Landslides

---

## 1. Executive Scientific Motivation

Across mountain transportation corridors, stream crossings rely on **culverts** (corrugated metal pipes and concrete box channels) to convey runoff beneath road embankments. During severe precipitation events, culverts frequently suffer catastrophic structural blowouts through a multi-hazard coupling:

```
                  [ Extreme Precipitation Event ]
                                │
       ┌────────────────────────┴────────────────────────┐
       ▼                                                 ▼
[ 2D Topographic Wetness (TWI) ]            [ Sentinel-1 InSAR Kinematics ]
Hollows saturate (m(x,y) -> 1.0)            Downslope creep (v_LOS < -10 mm/yr)
       │                                                 │
       ▼                                                 ▼
[ Pore Pressure: u = m * γ_w * z * cos²β ]  [ Kinematic Strain Softening ]
       │                                                 │
       └────────────────────────┬────────────────────────┘
                                ▼
             [ Terzaghi-Coulomb Slope Failure: FS < 1.0 ]
                                │ (Debris Flow Torrent)
                                ▼
       ═════════════════[ ROADWAY EMBANKMENT ]═════════════════
          [ CULVERT INLET ] ◄── Woody Debris Constriction (GRAIP)
       ────────────────────────────────────────────────────────
                                │
                                ▼
            💥 [ CATASTROPHIC ROAD EMBANKMENT BLOWOUT ]
```

1. **Topographic Moisture Convergence:** Regional satellite soil moisture (e.g. NASA SMAP) concentrates in convergent mountain hollows and riparian zones, driving local saturation ($m(x,y) \rightarrow 1.0$).
2. **Geotechnical Instability & InSAR Validation:** Elevated pore-water pressure ($u$) reduces effective normal stress ($\sigma' = \sigma - u$), triggering translational slope failure, validated observationally by **Sentinel-1 InSAR line-of-sight displacement velocities ($v_{\text{LOS}}$)**.
3. **Woody Debris & Hydraulic Surcharge:** Narrow culverts become choked with woody debris (USDA Forest Service GRAIP model), drastically reducing conveyance capacity and causing water to pool until the road embankment washes out.

This open-source framework automates the entire coupled pipeline from raw DEMs and road vectors to operational risk ranking.

---

## 2. Mathematical & Physical Formulation

### A. Geotechnical Slope Stability (Terzaghi-Coulomb Infinite Slope Model)
The Factor of Safety ($FS$) against translational shear failure along an inclined slip plane:

$$FS = \frac{c' + (\gamma_s - m(x,y) \gamma_w) z \cos^2\beta \tan\phi'}{\gamma_s z \sin\beta \cos\beta}$$

* $c'$: Effective soil cohesion ($\text{kPa}$)
* $\phi'$: Internal friction angle ($\text{degrees}$)
* $\beta$: Topographic hillslope gradient ($\text{degrees}$, derived from DEM finite differences)
* $z$: Soil mantle thickness down to failure plane ($\text{m}$)
* $\gamma_s, \gamma_w$: Soil bulk unit weight ($18.5 \text{ kN/m}^3$) and water unit weight ($9.81 \text{ kN/m}^3$)
* $m(x,y)$: **2D Spatially Distributed Hydrological Saturation Ratio** ($0.05 \le m \le 1.0$)

---

### B. Hyper-Resolution Soil Moisture (Beven-Kirkby TWI / SMAP Downscaling)
To emulate the physical downscaling logic of **SMAP-HydroBlocks** (Vergopolan et al., 2021), regional satellite soil moisture is downscaled using the Topographic Wetness Index:

$$\text{TWI} = \ln\left( \frac{a}{\tan\beta + \epsilon} \right)$$

Where $a = N_{\text{accum}} \cdot \Delta x$ is specific contributing drainage area per unit contour width. Convergent hollows concentrate antecedent saturation toward 100%, while steep ridges drain rapidly.

---

### C. Sentinel-1 InSAR Kinematic Observational Prior
Static slope equations only predict *theoretical* susceptibility. Satellite radar interferometry (**InSAR**) measures actual physical ground movement. The prior Factor of Safety is dynamically updated via an exponential kinematic strain-softening function:

$$v_{\text{excess}} = \max\Big(0, \; |v_{\text{LOS}}| - v_{\text{threshold}}\Big)$$

$$FS_{\text{insar}} = FS_{\text{prior}} \cdot \exp\left( -\frac{v_{\text{excess}}}{v_{\text{scale}}} \right)$$

Where $v_{\text{threshold}} = 3.5 \text{ mm/yr}$ (geodetic noise floor) and $v_{\text{scale}} = 18.0 \text{ mm/yr}$. Active downslope creep confirms ongoing slip-surface shear displacement.

---

### D. USDA Forest Service Woody Debris Inlet Plugging (GRAIP)
In steep forested catchments, culvert failure is dominated by woody debris and coarse bedload. The inlet constriction ratio is:

$$\chi = \frac{D_{\text{culvert}}}{W_{\text{bankfull}}} = \frac{D}{1.9 \cdot A_{\text{catchment}}^{0.38}}$$

The debris jamming probability $P_{\text{plug}}$ reduces effective hydraulic conveyance:

$$Q_{\text{effective}} = Q_{\text{capacity}} \cdot \Big( 1.0 - 0.65 \cdot P_{\text{plug}} \Big)$$

---

### E. Composite Vulnerability Index (CVI) & Blowout Multiplier
$$\text{CVI} = \Big[ 0.45 \cdot S_{\text{geotech}} + 0.45 \cdot S_{\text{hydraulic}} + 0.10 \cdot P_{\text{plug}} \cdot 100 \Big] \cdot W_{\text{road}} \cdot \Psi_{\text{blowout}} \cdot \Phi_{\text{insar}}$$

Where $\Psi_{\text{blowout}} = 1.40$ is the **Embankment Blowout Multiplier** triggered when hydraulic surcharge ($HR \ge 1.0$) coincides with geotechnical instability ($FS_{\text{insar}} \le 1.25$).

---

## 3. Package Architecture

```
culvert_slope_mini_project/
├── data/
│   ├── sample_generator.py      # High-relief mountain watershed & road vector generator
│   ├── sample_dem.tif           # 25m DEM GeoTIFF (EPSG:32630)
│   └── sample_roads.geojson     # Road network vector corridors
├── src/
│   ├── hydro_routing.py         # DEM conditioning, finite-difference slope, D8 flow accumulation
│   ├── road_intersection.py     # Road-stream spatial overlay & upstream catchment delineation
│   ├── soil_moisture_twi.py     # Beven-Kirkby TWI & SMAP hyper-resolution downscaling
│   ├── insar_prior.py           # Sentinel-1 InSAR LOS velocity field & kinematic FS updater
│   ├── slope_physics.py         # Terzaghi-Coulomb FS modeling & m_crit analytical solver
│   ├── hydraulic_capacity.py    # Peak storm runoff & Manning culvert flow capacity
│   ├── debris_hazard.py         # USDA Forest Service woody debris & sediment inlet plugging
│   ├── risk_engine.py           # Multi-hazard risk prioritization & blowout scoring (v2.0)
│   └── visualizer.py            # Interactive Folium Web GIS map & 4-panel diagnostic figures
├── notebooks/
│   └── 01_pipeline_walkthrough.ipynb  # Interactive step-by-step Jupyter demonstration
├── tests/
│   ├── test_hydro_routing.py    # D8 flow direction, gradient & accumulation tests
│   ├── test_slope_physics.py    # Analytical FS and moisture sensitivity tests
│   ├── test_hydraulic_capacity.py # Manning's conveyance & discharge scaling tests
│   ├── test_soil_moisture_twi.py# TWI and SMAP downscaling boundary tests
│   ├── test_insar_prior.py      # InSAR kinematic degradation & stationary limit tests
│   └── test_debris_hazard.py    # Bankfull channel width & plugging probability tests
├── output/
│   ├── culvert_risk_map.html    # Interactive Web GIS Folium map (with InSAR badges)
│   ├── culvert_risk_diagnostics.png # 4-panel 300 DPI scientific diagnostic figure
│   └── culvert_risk_assessment_summary.csv # Tabular engineering dataset
├── run_pipeline.py              # CLI driver with configurable parameters
├── requirements.txt             # Lightweight Python dependencies
├── pyproject.toml               # Python packaging metadata
└── LICENSE                      # MIT Open-Source License
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

### Run Full Advanced Pipeline
```bash
python run_pipeline.py --regional-saturation 0.70 --rainfall 75.0 --diameter 1.2
```

### Run Unit Tests
```bash
pytest tests/ -v
# 17 passed in ~5 seconds
```

---

## 5. Visual Outputs & Cartographic Products

### Interactive Web GIS Map (`output/culvert_risk_map.html`)
* **Color-coded risk classifications:** Green (Low) $\rightarrow$ Yellow (Moderate) $\rightarrow$ Orange (High) $\rightarrow$ Red (Critical Blowout).
* **Popup engineering cards:** Displays InSAR LOS velocity ($v_{\text{LOS}}$), kinematic activity status, TWI local saturation ratio ($m$), woody debris jamming probability, and hydraulic surcharge ratio ($HR$).

### 4-Panel Publication Diagnostic Figure (`output/culvert_risk_diagnostics.png`)
* **Panel A:** Factor of Safety degradation curves across antecedent soil saturation.
* **Panel B:** Sentinel-1 InSAR LOS velocity vs updated geotechnical stability ($FS_{\text{insar}}$).
* **Panel C:** Peak storm discharge vs nominal and debris-constricted pipe capacity.
* **Panel D:** Advanced Composite Vulnerability Index (CVI) ranking.

---

## 6. Scientific Foundations & Prior Art Citations

1. **USDA Forest Service GRAIP (Geomorphic Road Analysis and Inventory Package):**
   * Cissel, J. M., et al. (2014). *GRAIP-Lite: A tool for evaluating road sediment delivery and culvert failure risk across watershed scales*. USDA Forest Service Rocky Mountain Research Station.
   * Furniss, M. J., et al. (1998). *Response of road-stream crossings to large flood events in Washington, Oregon, and Northern California*. USDA Forest Service.
2. **Hyper-Resolution Hydrology & SMAP Satellite Downscaling:**
   * Vergopolan, N., et al. (2021). *SMAP-HydroBlocks: A 30-m satellite-based soil moisture dataset for hyper-resolution environmental modeling across the United States*. Scientific Data.
3. **Multi-Temporal InSAR for Slope Deformation:**
   * Aryee, P., & Kumi-Boateng, B. (2024). *Assessing Ground Deformation at Chirano Gold Mine Pit Slopes using Multi-Temporal Sentinel-1 SAR Interferometry*. UMaT Thesis.
4. **Topographic Wetness & Hillslope Dynamics:**
   * Beven, K. J., & Kirkby, M. J. (1979). *A physically based, variable contributing area model of basin hydrology*. Hydrological Sciences Bulletin.
   * Hobley, D. E. J., et al. (2017). *Creative computing with Landlab: an open-source library for building 2D numerical models of Earth-surface dynamics*. Earth Surface Dynamics.

---

## 7. License

Distributed under the **MIT License**. See `LICENSE` for details.
