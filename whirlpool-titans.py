# impactor_streamlit.py

import os
import math
import json
import time
import base64
from typing import Optional, Tuple, List

import streamlit as st
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from functools import lru_cache

# Try optional imports
try:
    from poliastro.twobody import Orbit
    from poliastro.bodies import Sun
    from astropy import units as u
    POLIASTRO_AVAILABLE = True
except ImportError:
    POLIASTRO_AVAILABLE = False

import plotly.graph_objects as go
import pydeck as pdk
from pyproj import Geod
from scipy import optimize

# --- Constants ---
G = 6.67430e-11
EARTH_RADIUS_M = 6371000.0
DEFAULT_DENSITY = 3000.0

NASA_NEO_BASE = "https://api.nasa.gov/neo/rest/v1"
USGS_EARTHQUAKE_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_EPQS = "https://epqs.nationalmap.gov/v1/json"

geod = Geod(ellps="WGS84")

# --- Utility & API helpers ---

def read_api_key() -> Optional[str]:
    key = os.getenv("NASA_API_KEY") or (st.secrets.get("NASA_API_KEY") if "NASA_API_KEY" in st.secrets else None)
    return key

@st.cache_data(ttl=3600)
def nasa_neo_lookup_by_id(neo_id: str, api_key: str) -> dict:
    url = f"{NASA_NEO_BASE}/neo/{neo_id}?api_key={api_key}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def nasa_neo_feed(start_date: str, end_date: str, api_key: str) -> dict:
    url = f"{NASA_NEO_BASE}/feed?start_date={start_date}&end_date={end_date}&api_key={api_key}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def usgs_earthquake_search(starttime: str, endtime: str, minmagnitude: float = 2.5,
                           latitude: Optional[float] = None, longitude: Optional[float] = None,
                           maxradiuskm: Optional[float] = None) -> dict:
    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": minmagnitude,
    }
    if latitude is not None and longitude is not None and maxradiuskm is not None:
        params.update({"latitude": latitude, "longitude": longitude, "maxradiuskm": maxradiuskm})
    r = requests.get(USGS_EARTHQUAKE_QUERY, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def usgs_elevation_point(lat: float, lon: float) -> dict:
    params = {"x": lon, "y": lat, "units": "Meters", "output": "json"}
    r = requests.get(USGS_EPQS, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# --- Physics / modeling helpers ---

def asteroid_mass_from_diameter(diameter_m: float, density: float = DEFAULT_DENSITY) -> float:
    r = diameter_m / 2.0
    volume = (4.0 / 3.0) * math.pi * (r ** 3)
    return volume * density

def kinetic_energy_joules(mass_kg: float, velocity_m_s: float) -> float:
    return 0.5 * mass_kg * (velocity_m_s ** 2)

def joules_to_megatons_tnt(j: float) -> float:
    return j / (4.184e15)

def crater_diameter_estimate_simple(energy_j: float) -> float:
    k = 1.8
    return k * (energy_j ** (1.0 / 3.4))

def seismic_magnitude_estimate(energy_j: float) -> float:
    f = 1e-4
    Es = f * energy_j
    if Es <= 0:
        return 0.0
    Mw = (math.log10(Es) - 4.8) / 1.5
    return Mw

def render_orbit_3d_plot(positions: np.ndarray) -> go.Figure:
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines+markers",
                              name="Asteroid path", line=dict(width=3), marker=dict(size=2)))
    # Earth sphere
    uvals = np.linspace(0, 2 * np.pi, 40)
    vvals = np.linspace(0, np.pi, 20)
    x_s = EARTH_RADIUS_M * np.outer(np.cos(uvals), np.sin(vvals))
    y_s = EARTH_RADIUS_M * np.outer(np.sin(uvals), np.sin(vvals))
    z_s = EARTH_RADIUS_M * np.outer(np.ones_like(uvals), np.cos(vvals))
    fig.add_trace(go.Surface(x=x_s, y=y_s, z=z_s, showscale=False, opacity=0.6, name="Earth"))
    fig.update_layout(scene=dict(aspectmode="auto"), margin=dict(l=0, r=0, t=0, b=0))
    return fig

def latlon_from_cartesian(r_vec: np.ndarray) -> Tuple[float, float]:
    x, y, z = r_vec
    lon = math.degrees(math.atan2(y, x))
    hyp = math.sqrt(x*x + y*y)
    lat = math.degrees(math.atan2(z, hyp))
    return lat, lon

# --- Streamlit UI & app logic ---

def sidebar_controls():
    st.sidebar.title("Controls")
    api_key = st.sidebar.text_input("NASA API Key", value=read_api_key() or "")
    mode = st.sidebar.radio("Mode", ["Explore NEOs", "Simulate Impact", "Defend Earth"])
    return api_key.strip() or None, mode

def display_neo_summary(neo: dict):
    st.subheader(f"NEO Summary: {neo.get('name')}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("ID / Designation")
        st.write(neo.get("id"))
        st.write("Absolute Magnitude (H)")
        st.write(neo.get("absolute_magnitude_h"))
        diam = neo.get("estimated_diameter", {}).get("meters", {})
        st.write("Diameter (m) range")
        st.write(f"{diam.get('estimated_diameter_min', '?')} – {diam.get('estimated_diameter_max', '?')}")
    with col2:
        st.write("Potentially Hazardous?")
        st.write(neo.get("is_potentially_hazardous_asteroid", False))
        st.write("Orbit parameters")
        orb = neo.get("orbital_data", {})
        st.write(f"SMA (au): {orb.get('semi_major_axis')}")
        st.write(f"Eccentricity: {orb.get('eccentricity')}")
        st.write(f"Inclination (°): {orb.get('inclination')}")
    with col3:
        st.write("Next close approaches")
        cad = neo.get("close_approach_data", [])
        for c in cad[:2]:
            st.write(f"- date: {c.get('close_approach_date')}, v: {c.get('relative_velocity', {}).get('kilometers_per_second')} km/s, miss: {c.get('miss_distance', {}).get('kilometers')} km")

def run_simulation_mode(neo: dict):
    st.header("Simulation / Impact Mode")

    diam_guess = (neo.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_min", 50.0) +
                  neo.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_max", 50.0)) / 2.0
    diameter_m = st.number_input("Asteroid diameter (m)", value=float(diam_guess), min_value=0.1)
    density = st.number_input("Asteroid density (kg/m³)", value=DEFAULT_DENSITY, min_value=500.0)
    mass = asteroid_mass_from_diameter(diameter_m, density)

    cad = neo.get("close_approach_data", [])
    if cad:
        v_km_s = float(cad[0].get("relative_velocity", {}).get("kilometers_per_second", 20.0))
    else:
        v_km_s = 20.0
    v_m_s = v_km_s * 1000.0

    st.write(f"Estimated incoming velocity: {v_km_s:.3f} km/s")

    energy = kinetic_energy_joules(mass, v_m_s)
    mt = joules_to_megatons_tnt(energy)
    crater = crater_diameter_estimate_simple(energy)
    Mw = seismic_magnitude_estimate(energy)

    st.metric("Mass (kg)", f"{mass:,.0f}")
    st.metric("Energy (J)", f"{energy:.3e}")
    st.metric("Equivalent (megatons TNT)", f"{mt:.3f}")
    st.write(f"Estimated crater diameter: ~{crater:.0f} m")
    st.write(f"Seismic-equivalent magnitude (rough): Mw {Mw:.2f}")

    st.markdown("---")
    st.subheader("Mitigation by delta-v")

    time_until_days = st.number_input("Time until encounter (days)", value=30.0)
    dv = st.number_input("Delta-v to apply (m/s)", value=0.0, min_value=0.0, max_value=100.0)
    apply_before = st.number_input("Days before encounter to apply delta-v", value=5.0, min_value=0.0, max_value=time_until_days)
    time_s = time_until_days * 86400.0
    apply_s = apply_before * 86400.0
    effective = max(0.0, time_s - apply_s)
    miss_offset = dv * effective

    st.write(f"Estimated lateral deflection at encounter: {miss_offset/1000.0:.2f} km")

def defend_earth_mode(neo: dict):
    st.header("Defend Earth — Game Mode")

    diam_guess = (neo.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_min", 50.0) +
                  neo.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_max", 50.0)) / 2.0
    diameter_m = st.number_input("Asteroid diameter (m)", value=float(diam_guess), min_value=0.1)
    density = st.number_input("Asteroid density (kg/m³)", value=DEFAULT_DENSITY, min_value=500.0)
    mass = asteroid_mass_from_diameter(diameter_m, density)

    st.write(f"Asteroid mass estimate: {mass:,.0f} kg")

    imp_mass = st.number_input("Kinetic impactor mass (kg)", value=1000.0, min_value=1.0)
    imp_speed = st.number_input("Impactor speed (km/s)", value=10.0, min_value=1.0) * 1000.0
    days_until = st.number_input("Days until encounter", value=180.0)
    days_before = st.slider("Days before encounter to impact", min_value=0.0, max_value=days_until, value=30.0)
    t_until = days_until * 86400.0
    t_before = days_before * 86400.0

    dv = (imp_mass * imp_speed) / mass
    defl = dv * max(0.0, t_until - t_before)

    st.write(f"Estimated delta-v imparted: {dv:.6f} m/s")
    st.write(f"Estimated deflection at encounter: {defl/1000.0:.2f} km")

    # Score
    score = int(min(100, defl / 1000.0))
    st.metric("Mission Score", f"{score}/100")

# --- Main app ---

def main():
    st.title("Impactor-2025 Simulator & Visualizer")
    api_key, mode = sidebar_controls()
    if not api_key:
        st.warning("Please enter your NASA API key to fetch asteroid data.")
        return

    neo_id = st.text_input("Enter NEO designation or ID", value="Impactor-2025")
    neo = None
    try:
        neo = nasa_neo_lookup_by_id(neo_id, api_key)
    except Exception as e:
        st.error(f"Failed to fetch NEO data: {e}")
        return

    display_neo_summary(neo)

    if mode == "Explore NEOs":
        st.subheader("Explore NEO feed (next 7 days)")
        try:
            now = datetime.utcnow().date()
            feed = nasa_neo_feed(now.isoformat(), (now + timedelta(days=7)).isoformat(), api_key)
            objs = []
            for d, arr in feed.get("near_earth_objects", {}).items():
                for obj in arr:
                    objs.append({
                        "name": obj.get("name"),
                        "id": obj.get("id"),
                        "close_date": d,
                        "is_hazardous": obj.get("is_potentially_hazardous_asteroid", False)
                    })
            df = pd.DataFrame(objs)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error fetching feed: {e}")

    elif mode == "Simulate Impact":
        run_simulation_mode(neo)

    elif mode == "Defend Earth":
        defend_earth_mode(neo)

if __name__ == "__main__":
    main()
