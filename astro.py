*** FILE: requirements.txt ***
streamlit>=1.25
requests>=2.28
numpy>=1.24
scipy>=1.10
pandas>=2.0
plotly>=5.0
skyfield>=1.47
astropy>=5.0
folium>=0.13
streamlit-folium>=0.10
pyproj>=3.6
shapely>=2.0
python-dotenv>=1.0
rasterio>=1.3; platform_system != "Windows"
Pillow>=9.5

*** FILE: .env.example ***
NASA_API_KEY=DEMO_KEY
USGS_API_ENDPOINT=[https://earthquake.usgs.gov/fdsnws/event/1/query](https://earthquake.usgs.gov/fdsnws/event/1/query)

*** FILE: vercel.json ***
{
"builds": [
{ "src": "streamlit_app.py", "use": "@vercel/python" }
],
"routes": [
{ "src": "/(.*)", "dest": "streamlit_app.py" }
]
}

*** FILE: Procfile ***
web: streamlit run streamlit_app.py

*** FILE: README.md ***

# Impactor-2025 — Asteroid Impact Simulator (Streamlit)

# Files


* data_fetch.py     — NASA / JPL / USGS data wrappers
* simulation.py     — impact physics, crater scaling, deflection models
* orbital.py        — orbit propagation helpers (skyfield optional)
* usgs.py           — USGS queries + simple inundation heuristics
* visuals.py        — map & 3D plotting helpers
* utils.py          — utility helpers

Create `.env` from `.env.example` and set `NASA_API_KEY`. Install requirements and run:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

*** FILE: utils.py ***
import os
import math
import json
from dotenv import load_dotenv

def load_env(path: str = ".env"):
if os.path.exists(path):
load_dotenv(path)

def jprint(obj):
print(json.dumps(obj, indent=2, default=str))

def tnt_megatons(joules: float) -> float:
return joules / 4.184e15

def clamp(v, a, b):
return max(a, min(b, v))

def safe_get(d, k, default=None):
return d.get(k, default) if isinstance(d, dict) else default

def fmt(x):
if x is None:
return "N/A"
try:
if abs(x) >= 1e6 or abs(x) < 1e-3:
return f"{x:.3e}"
return f"{x:,.3f}"
except Exception:
return str(x)

*** FILE: data_fetch.py ***
import os
import requests
from typing import Optional, Dict, Any
from utils import safe_get

NASA_API = "[https://api.nasa.gov](https://api.nasa.gov)"
JPL_SBDB = "[https://ssd-api.jpl.nasa.gov/sbdb.api](https://ssd-api.jpl.nasa.gov/sbdb.api)"
USGS_EQ_API = os.getenv("USGS_API_ENDPOINT", "[https://earthquake.usgs.gov/fdsnws/event/1/query](https://earthquake.usgs.gov/fdsnws/event/1/query)")

def nasa_api_wrapper(path: str, params: dict, api_key: str) -> dict:
params = params.copy()
if api_key:
params["api_key"] = api_key
r = requests.get(f"{NASA_API}{path}", params=params, timeout=20)
r.raise_for_status()
return r.json()

def fetch_neo_by_designation(designation: str, api_key: Optional[str]=None) -> Optional[Dict[str,Any]]:
"""
Try NASA NEO API first, fallback to JPL SBDB.
"""
# Try NASA NEO REST
if api_key:
try:
path = f"/neo/rest/v1/neo/{designation}"
r = requests.get(f"{NASA_API}{path}", params={"api_key": api_key}, timeout=20)
if r.status_code == 200:
return r.json()
except Exception:
pass
# Fallback to JPL SBDB search
try:
r = requests.get(JPL_SBDB, params={"sstr": designation, "phys-par": "1"}, timeout=20)
r.raise_for_status()
data = r.json()
return data
except Exception:
return None

def fetch_neo_search(query: str, api_key: Optional[str]=None):
# Basic wrapper: use NASA if available, otherwise SBDB
return fetch_neo_by_designation(query, api_key=api_key)

def query_usgs_earthquakes(lat: float, lon: float, maxradiuskm: int = 500, starttime: str = "2020-01-01", endtime: str = "2025-12-31"):
params = {
"format": "geojson",
"latitude": lat,
"longitude": lon,
"maxradiuskm": maxradiuskm,
"starttime": starttime,
"endtime": endtime,
"limit": 200
}
r = requests.get(USGS_EQ_API, params=params, timeout=20)
r.raise_for_status()
return r.json()

*** FILE: simulation.py ***
import math
from typing import Tuple
from utils import tnt_megatons

# Basic physical helpers and impact scaling relationships

def compute_mass_energy(diameter_m: float, density_kg_m3: float, velocity_m_s: float) -> Tuple[float,float]:
"""
Compute mass (kg) and kinetic energy (J).
diameter_m: meters
density_kg_m3: kg/m3
velocity_m_s: m/s
"""
r = diameter_m / 2.0
volume = (4.0/3.0) * math.pi * (r**3)
mass = volume * density_kg_m3
energy = 0.5 * mass * velocity_m_s**2
return mass, energy

def crater_diameter_transient(energy_joule: float, target_density=2500.0) -> float:
"""
Very rough transient crater scaling based on energy.
D_t (m) = k * E^(1/3.4)
"""
k = 1.5
D = k * (max(energy_joule, 1.0) ** (1.0/3.4))
# enforce plausible lower bound
if D < 2.0:
D = 2.0
return D

def classify_impact(diameter_m: float, energy_joule: float) -> str:
"""
Heuristic classification: airburst vs ground impact
"""
if diameter_m < 50 and energy_joule < 1e15:
return "airburst"
return "ground impact"

def deflection_approx_shift(impact_lat: float, impact_lon: float, delta_v: float, lead_time_days: float):
"""
Linearized shift approximation: shift (deg) ~ delta_v * lead_time_seconds / 111000
Returns new_lat, new_lon
"""
lead_seconds = lead_time_days * 86400.0
lateral_m = delta_v * lead_seconds
deg_lat = lateral_m / 111000.0
deg_lon = lateral_m / (111000.0 * max(0.0001, math.cos(math.radians(impact_lat))))
new_lat = impact_lat + deg_lat
new_lon = impact_lon + deg_lon
if new_lon > 180: new_lon -= 360
if new_lon < -180: new_lon += 360
return new_lat, new_lon

*** FILE: orbital.py ***
import math
from typing import Tuple, List, Optional
try:
from skyfield.api import load, Loader, EarthSatellite, Topos
SKYFIELD_AVAILABLE = True
except Exception:
SKYFIELD_AVAILABLE = False
import numpy as np

def propagate_orbit_approx(a_au=1.2, b_au=0.9, steps=360):
"""
Parametric ellipse for visualization (AU units)
"""
thetas = np.linspace(0, 2*np.pi, steps)
coords = []
for th in thetas:
x = a_au * math.cos(th)
y = b_au * math.sin(th)
z = 0.02 * math.sin(2*th)
coords.append((x,y,z))
return coords

def build_orbit_trace_from_elements(elements: dict, steps=360):
"""
Very simple build: derive ellipse parameters from a (semi-major) and e and return coords.
elements expected to have keys: a (AU), e (eccentricity), i (deg), om (Ω), w (ω)
"""
a = float(elements.get("a", 1.2))
e = float(elements.get("e", 0.1))
b = a * math.sqrt(1 - e*e)
coords = propagate_orbit_approx(a_au=a, b_au=b, steps=steps)
return coords

def propagate_orbit_skyfield_from_elements(elements: dict, days=365, steps=360):
"""
If Skyfield present and elements are valid, attempt more realistic propagation.
This function is best-effort and will fallback to approximate orbit if elements missing.
"""
if not SKYFIELD_AVAILABLE:
return propagate_orbit_approx(steps=steps)
# For now use approximate method; integrating Keplerian properly requires more code.
return propagate_orbit_approx(steps=steps)

*** FILE: usgs.py ***
from typing import Dict, Any
import numpy as np
import rasterio
import os

def query_earthquakes(lat: float, lon: float, radius_km: int = 500):
"""
Wrapper around data_fetch.query_usgs_earthquakes expected to be used in Streamlit app.
Kept lightweight here.
"""
# This function will be called with data_fetch's function; placeholder for modularity
return None

def estimate_inundation_from_dem(lat: float, lon: float, crater_diameter_m: float, dem_path: str = None) -> Dict[str,Any]:
"""
Very rough inundation estimator using DEM: look up elevation at impact location and estimate radial inundation.
Requires rasterio and a DEM path.
"""
if dem_path is None or not os.path.exists(dem_path):
# fallback heuristic
if crater_diameter_m > 500:
return {"inundation_km": min(200, crater_diameter_m/100.0*5)}
return {"inundation_km": 0}
try:
with rasterio.open(dem_path) as ds:
forval = list(ds.sample([(lon, lat)]))
elev = forval[0][0] if forval and forval[0] is not None else None
if elev is None:
return {"inundation_km": 0}
# simplistic: lower elevation => larger inundation
if elev < 5 and crater_diameter_m > 200:
return {"inundation_km": min(300, crater_diameter_m/50.0)}
return {"inundation_km": 0}
except Exception:
return {"inundation_km": 0}

*** FILE: visuals.py ***
import folium
from folium.plugins import FloatImage
import plotly.graph_objects as go

def folium_impact_map(lat: float, lon: float, crater_m: float, inundation_km: float = 0.0, markers: list = None):
m = folium.Map(location=[lat, lon], zoom_start=3, tiles="CartoDB positron")
# crater circle
folium.Circle(location=[lat, lon],
radius=max(100, crater_m/2),
color="crimson", fill=True, fill_opacity=0.4,
tooltip=f"Approx crater diameter: {crater_m:.1f} m").add_to(m)
# inundation ring
if inundation_km and inundation_km > 0:
folium.Circle(location=[lat, lon],
radius=inundation_km*1000,
color="blue", fill=False, opacity=0.5,
tooltip=f"Estimated inundation radius: {inundation_km} km").add_to(m)
if markers:
for mm in markers:
folium.Marker(location=[mm[0], mm[1]], popup=mm[2] if len(mm)>2 else None).add_to(m)
return m

def plotly_orbit_3d(coords: list, title="Orbit visualization"):
xs = [c[0] for c in coords]
ys = [c[1] for c in coords]
zs = [c[2] for c in coords]
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="Asteroid"))
# Earth marker at origin approx
fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=6, color="blue"), name="Earth"))
fig.update_layout(scene=dict(xaxis_title='X (AU)', yaxis_title='Y (AU)', zaxis_title='Z (AU)'), height=600, title=title)
return fig

*** FILE: streamlit_app.py ***
#!/usr/bin/env python3
import os
import math
import json
from datetime import datetime, timedelta
import streamlit as st
from streamlit_folium import st_folium

from utils import load_env, fmt, tnt_megatons
from data_fetch import fetch_neo_by_designation, query_usgs_earthquakes, nasa_api_wrapper
from simulation import compute_mass_energy, crater_diameter_transient, classify_impact, deflection_approx_shift
from orbital import propagate_orbit_approx, build_orbit_trace_from_elements, propagate_orbit_skyfield_from_elements
from visuals import folium_impact_map, plotly_orbit_3d

load_env()

NASA_API_KEY = os.getenv("NASA_API_KEY", "")

st.set_page_config(layout="wide", page_title="Impactor-2025")
st.title("Impactor-2025 — Asteroid Impact Simulator")

# Sidebar inputs and presets

st.sidebar.header("Scenario & Data")
data_source = st.sidebar.selectbox("Data source", ["Manual input", "NASA/JPL lookup"])
designation = ""
if data_source != "Manual input":
designation = st.sidebar.text_input("Designation / Name (e.g., 2025 PDC)", value="Impactor-2025")
if st.sidebar.button("Fetch NEO"):
neo = fetch_neo_by_designation(designation, api_key=NASA_API_KEY)
if neo:
st.session_state['neo_data'] = neo
st.sidebar.success("NEO data fetched")
else:
st.sidebar.error("Failed to fetch NEO data")

st.sidebar.markdown("### Asteroid Parameters")
diameter_m = st.sidebar.number_input("Diameter (m)", min_value=1.0, value=120.0, step=1.0)
velocity_kms = st.sidebar.number_input("Velocity (km/s)", min_value=1.0, value=20.0, step=0.1)
density = st.sidebar.number_input("Density (kg/m³)", min_value=500.0, value=3000.0, step=10.0)

st.sidebar.markdown("### Impact location")
impact_lat = st.sidebar.slider("Latitude", -85.0, 85.0, 0.0)
impact_lon = st.sidebar.slider("Longitude", -180.0, 180.0, 0.0)

st.sidebar.markdown("### Mitigation")
enable_deflect = st.sidebar.checkbox("Enable deflection simulation", value=False)
delta_v_ms = st.sidebar.slider("Δv (m/s)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
lead_days = st.sidebar.number_input("Lead days before impact", min_value=0, value=365)

st.sidebar.markdown("### Visualization")
use_skyfield = st.sidebar.checkbox("Use Skyfield propagation if elements available", value=True)
prop_days = st.sidebar.slider("Visualization propagation days", 7, 365, value=120)

# Main UI

col1, col2 = st.columns([1,1])
with col1:
st.header("Inputs")
st.write(f"Diameter: **{diameter_m} m**, Velocity: **{velocity_kms} km/s**, Density: **{density} kg/m³**")
st.write(f"Impact coords: **{impact_lat:.3f}°, {impact_lon:.3f}°**")
run = st.button("Run full simulation")
with col2:
st.header("Context")
st.write("This app uses approximations suitable for education and preliminary analysis. Replace models with high-fidelity solvers for operational use.")

if run:
# compute mass & energy
mass, energy_j = compute_mass_energy(diameter_m, density, velocity_kms*1000.0)
energy_mt = tnt_megatons(energy_j)
impact_type = classify_impact(diameter_m, energy_j)
crater_m = crater_diameter_transient(energy_j)
st.metric("Mass (kg)", fmt(mass))
st.metric("Impact energy (Mt TNT)", fmt(energy_mt))
st.metric("Estimated transient crater diameter (m)", fmt(crater_m))
st.write("Impact classification:", impact_type)

```
# orbital propagation / visualization
coords = None
if 'neo_data' in st.session_state and use_skyfield:
    try:
        coords = propagate_orbit_skyfield_from_elements(st.session_state['neo_data'], days=prop_days)
    except Exception:
        coords = None
if coords is None:
    coords = propagate_orbit_approx(steps=360)

st.subheader("3D orbit (approx)")
fig = plotly_orbit_3d(coords, title="Asteroid trajectory (approx)")
st.plotly_chart(fig, use_container_width=True)

# deflection simulation
if enable_deflect and delta_v_ms > 0:
    new_lat, new_lon = deflection_approx_shift(impact_lat, impact_lon, delta_v_ms, lead_days)
    st.write(f"Approx new impact coordinates after Δv {delta_v_ms:.2f} m/s applied {lead_days} days prior: {new_lat:.3f}, {new_lon:.3f}")
    markers = [(impact_lat, impact_lon, "Original impact"), (new_lat, new_lon, "Post-deflection impact")]
    inundation = None
    try:
        inundation = None  # placeholder: call DEM module if available
    except Exception:
        inundation = None
    m = folium_impact_map(impact_lat, impact_lon, crater_m, inundation_km=0, markers=markers)
else:
    m = folium_impact_map(impact_lat, impact_lon, crater_m, inundation_km=0)

st.subheader("Impact map")
st_folium(m, width=900, height=500)

# USGS context
st.subheader("USGS recent earthquakes near impact site (sample)")
try:
    # simple query using data_fetch helper
    from data_fetch import query_usgs_earthquakes
    usgs = query_usgs_earthquakes(impact_lat, impact_lon)
    # parse features -> table
    feats = usgs.get("features", []) if usgs else []
    rows = []
    for f in feats[:50]:
        p = f.get("properties", {})
        place = p.get("place")
        mag = p.get("mag")
        time = p.get("time")
        rows.append({"place": place, "magnitude": mag, "time": datetime.utcfromtimestamp(time/1000.0).isoformat() if time else None})
    st.table(rows[:20])
except Exception:
    st.write("USGS query not available.")

# export results
result = {
    "timestamp": datetime.utcnow().isoformat(),
    "diameter_m": diameter_m,
    "velocity_kms": velocity_kms,
    "density": density,
    "mass_kg": mass,
    "energy_j": energy_j,
    "energy_mt": energy_mt,
    "crater_m": crater_m,
    "impact_lat": impact_lat,
    "impact_lon": impact_lon,
    "deflection_enabled": bool(enable_deflect),
    "delta_v_ms": delta_v_ms,
    "lead_days": lead_days
}
st.download_button("Download simulation JSON", data=json.dumps(result, indent=2), file_name="impactor2025_result.json", mime="application/json")
```

*** END OF FILE BUNDLE ***
