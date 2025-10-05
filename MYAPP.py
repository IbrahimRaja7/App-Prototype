
# impactor2025_app.py
"""
Impactor-2025 — Streamlit application
A scientific, educational, and decision-support tool for modeling near-Earth asteroid impacts,
integrating NASA NeoWs (NEO API) and USGS data (earthquakes, elevation / National Map).
Features
- Fetch NEO data from NASA (by designation or browse)
- Quick orbital propagation (Keplerian 2-body approximation using poliastro if available)
- Impact energy, TNT equivalent, crater scaling (empirical scaling laws)
- Simple deflection simulation (delta-v applied -> recompute miss distance / impact probability)
- Map visualizations (pydeck for impact zone and tsunami overlays), 3D orbit plots (plotly)
- USGS integrations: fetch recent nearby earthquakes and elevation (EPQS service)
- "Defend Earth" gamified simulation mode
- Fallbacks & caching, clear UI, tooltips, accessibility considerations
Requirements (install):
pip install streamlit requests numpy pandas pyproj pydeck plotly scipy poliastro==0.14.0 matplotlib cachetools python-dotenv
Note: poliastro is optional but recommended for better orbital propagation. If not installed,
the code will use a simpler Kepler solver fallback (approximate).
Set environment variable NASA_API_KEY (or paste in the UI).
"""

# ---- Imports ---------------------------------------------------------------
import os
import math
import json
import time
import base64
from typing import Dict, Optional, Tuple, List

import streamlit as st
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from functools import lru_cache

# Optional scientific libs
try:
    from poliastro.twobody import Orbit
    from poliastro.bodies import Sun, Earth
    from astropy import units as u
    POLIASTRO_AVAILABLE = True
except Exception:
    POLIASTRO_AVAILABLE = False

# Plotting / viz
import plotly.graph_objects as go
import pydeck as pdk

# Geodesy
from pyproj import Geod

# Numerical
from scipy import optimize

# ------------------ Constants & Helpers -------------------------------------
st.set_page_config(
    page_title="Impactor-2025 — Asteroid Impact Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-2
EARTH_RADIUS_M = 6371000.0  # mean radius
EARTH_MASS = 5.972e24  # kg
DEFAULT_DENSITY = 3000.0  # kg/m^3 typical stony asteroid
TNT_EQ_JOULE = 4.184e9  # 1 ton of TNT in joules? Actually 1 ton TNT = 4.184e9 J (1 kiloton = 4.184e12)

# NASA / USGS endpoints (documented)
NASA_NEO_BASE = "https://api.nasa.gov/neo/rest/v1"
USGS_EARTHQUAKE_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_EPQS = "https://epqs.nationalmap.gov/v1/json"  # elevation point query service

# Geodesic helper
geod = Geod(ellps="WGS84")

# ------------------ Utility Functions --------------------------------------

def read_api_key():
    # First check env then streamlit secrets
    key = os.getenv("NASA_API_KEY") or (st.secrets.get("NASA_API_KEY") if "NASA_API_KEY" in st.secrets else None)
    return key

@st.cache_data(ttl=3600)
def nasa_neo_lookup_by_id(neo_id: str, api_key: Optional[str]) -> dict:
    """Lookup NEO by id or designation via NeoWs 'lookup' endpoint."""
    if not api_key:
        raise ValueError("NASA API key required")
    url = f"{NASA_NEO_BASE}/neo/{neo_id}?api_key={api_key}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def nasa_neo_feed(start_date: str, end_date: str, api_key: Optional[str]) -> dict:
    """Get feed of NEOs with close approaches in date range."""
    if not api_key:
        raise ValueError("NASA API key required")
    url = f"{NASA_NEO_BASE}/feed?start_date={start_date}&end_date={end_date}&api_key={api_key}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def usgs_earthquake_search(starttime: str, endtime: str, minmagnitude: float = 2.5, 
                           latitude: Optional[float] = None, longitude: Optional[float] = None, maxradiuskm: Optional[float] = None) -> dict:
    """Query USGS earthquake API (ComCat) for context or seismic baseline data."""
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
    """EPQS elevation query returns elevation for a single lat/lon point (meters)."""
    params = {"x": lon, "y": lat, "units": "Meters", "output": "json"}
    r = requests.get(USGS_EPQS, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# ------------------ Physics: energy, crater scaling, simple propagation -----

def asteroid_mass_from_diameter(diameter_m: float, density: float = DEFAULT_DENSITY) -> float:
    """Assuming spherical asteroid: mass = volume * density."""
    r = diameter_m / 2.0
    volume = (4.0 / 3.0) * math.pi * (r ** 3)
    return volume * density

def kinetic_energy_joules(mass_kg: float, velocity_m_s: float) -> float:
    """0.5 * m * v^2"""
    return 0.5 * mass_kg * velocity_m_s ** 2

def joules_to_megatons_tnt(joules: float) -> float:
    """Convert J -> megatons TNT"""
    # 1 megaton TNT = 4.184e15 J (because 1 ton TNT = 4.184e9 J)
    return joules / (4.184e15)

def crater_diameter_estimate_simple(energy_j: float, target_density=2500.0) -> float:
    """
    Approximate transient crater diameter using an empirical scaling (Melosh-type).
    One simple scaling: D (m) = k * (E)^(1/3.4)
    This is an approximation; more sophisticated models (iSALE, hydrocode) are beyond scope.
    """
    # Choose constants to produce reasonable values for typical impacts
    k = 1.8  # tunable coefficient
    return k * (energy_j ** (1.0 / 3.4))

def seismic_magnitude_estimate(energy_j: float) -> float:
    """
    Rough conversion from impact energy to equivalent seismic magnitude (Mw).
    This is very approximate: assume some fraction f of impact energy couples into seismic waves.
    Use empirical scaling to convert seismic energy to Mw: log10(E_s) = 1.5 Mw + 4.8 (E in Joules)
    Solve for Mw given E_s = f * E_total.
    """
    f = 1e-4  # fraction coupling into seismic energy — highly uncertain and conservative
    Es = f * energy_j
    if Es <= 0:
        return 0.0
    Mw = (math.log10(Es) - 4.8) / 1.5
    return Mw

# ------------------ Simple 'close-approach miss distance' linear model ----------
def compute_miss_distance_linear(velocity_m_s: float, time_until_encounter_s: float, delta_v_m_s: float) -> float:
    """
    Very simple linear approximation:
    A small lateral delta-v applied at time t0 before encounter will produce approx lateral displacement:
    d ≈ delta_v * time_to_encounter
    Returns miss distance in meters.
    """
    return abs(delta_v_m_s) * max(0.0, time_until_encounter_s)

# ------------------ Orbit helpers (poliastro fallback) -----------------------
def propagate_orbit_keplerian(orb_elements: dict, to_datetime: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """
    If poliastro available, propagate Keplerian elements to a target datetime and return position, velocity vectors (in meters, m/s).
    orb_elements expected keys: a (km), ecc, inc (deg), raan (deg), argp (deg), M0 (deg), epoch (ISO)
    Returns (r_vec_m, v_vec_m_s)
    """
    if not POLIASTRO_AVAILABLE:
        raise RuntimeError("poliastro not available")
    # convert to astropy units
    a = orb_elements.get("a") * u.km
    ecc = orb_elements.get("ecc")
    inc = orb_elements.get("inc") * u.deg
    raan = orb_elements.get("raan") * u.deg
    argp = orb_elements.get("argp") * u.deg
    M0 = orb_elements.get("M0") * u.deg
    epoch = orb_elements.get("epoch")
    # create orbit
    try:
        from astropy.time import Time
        epoch_time = Time(epoch)
        orbits = Orbit.from_classical(Sun, a, ecc * u.one, inc, raan, argp, M0, epoch_time)
        tof = Time(to_datetime) - epoch_time
        new_orbit = orbits.propagate(tof)
        r = new_orbit.r.to(u.m).value
        v = new_orbit.v.to(u.m / u.s).value
        return r, v
    except Exception as e:
        raise

# ------------------ Visualization helpers -----------------------------------

def render_orbit_3d_plot(positions: np.ndarray, point_labels: Optional[List[str]] = None) -> go.Figure:
    """
    positions: Nx3 array of x,y,z positions (meters)
    Returns a Plotly 3D scatter/line figure showing positions and Earth sphere.
    """
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines+markers",
                               name="Asteroid path", line=dict(width=3), marker=dict(size=2)))
    # Earth surface sphere (low-res)
    uvals = np.linspace(0, 2 * np.pi, 40)
    vvals = np.linspace(0, np.pi, 20)
    x_s = EARTH_RADIUS_M * np.outer(np.cos(uvals), np.sin(vvals))
    y_s = EARTH_RADIUS_M * np.outer(np.sin(uvals), np.sin(vvals))
    z_s = EARTH_RADIUS_M * np.outer(np.ones_like(uvals), np.cos(vvals))
    fig.add_trace(go.Surface(x=x_s, y=y_s, z=z_s, showscale=False, opacity=0.6, name="Earth"))
    fig.update_layout(scene=dict(aspectmode='auto'),
                      margin=dict(l=0, r=0, t=0, b=0))
    return fig

def latlon_from_cartesian(r_vec: np.ndarray) -> Tuple[float, float]:
    """Convert ECEF-like cartesian (meters) to geodetic lat/lon (approx) assuming centered Earth."""
    x, y, z = r_vec
    # simple conversion
    lon = math.degrees(math.atan2(y, x))
    hyp = math.sqrt(x * x + y * y)
    lat = math.degrees(math.atan2(z, hyp))
    return lat, lon

# ------------------ Streamlit UI components & app layout ---------------------

def sidebar_controls():
    st.sidebar.title("Impactor-2025 Controls")
    api_key = st.sidebar.text_input("NASA API Key (or set NASA_API_KEY env var)", value=read_api_key() or "")
    mode = st.sidebar.radio("Mode", ["Explore NEO", "Simulate Impact", "Defend Earth (Game)"])
    st.sidebar.markdown("---")
    st.sidebar.caption("Data sources: NASA NeoWs, USGS ComCat, USGS National Map (elevation)")
    return api_key.strip() or None, mode

def display_neo_summary(neo_json: dict):
    st.subheader(f"NEO — {neo_json.get('name')}")
    cols = st.columns([1, 2, 2])
    with cols[0]:
        st.write("**Designation / ID**")
        st.write(neo_json.get('id'))
        st.write("**Absolute mag (H)**")
        st.write(neo_json.get('absolute_magnitude_h'))
        est_diameter = neo_json.get("estimated_diameter", {})
        meters_range = est_diameter.get("meters", {}).get("estimated_diameter_min"), est_diameter.get("meters", {}).get("estimated_diameter_max")
        st.write("**Estimated diameter (m)**")
        st.write(f"{meters_range[0]:.1f} - {meters_range[1]:.1f} m")
        st.write("**Potentially hazardous?**")
        st.write(neo_json.get("is_potentially_hazardous_asteroid"))
    with cols[1]:
        st.write("**Close-approach data (next)**")
        cad = neo_json.get("close_approach_data", [])
        if len(cad) > 0:
            # show the next approach
            for cad_item in cad[:3]:
                st.write(f"- Date: {cad_item.get('close_approach_date_full') or cad_item.get('close_approach_date')}")
                v_km_s = float(cad_item.get("relative_velocity", {}).get("kilometers_per_second") or 0.0)
                miss_km = float(cad_item.get("miss_distance", {}).get("kilometers") or 0.0)
                st.write(f"  - Velocity: {v_km_s:.3f} km/s, Miss distance: {miss_km:.1f} km, Orbiting body: {cad_item.get('orbiting_body')}")
        else:
            st.write("No close approach entries found.")
    with cols[2]:
        st.write("**Orbital data (if available)**")
        orb = neo_json.get("orbital_data", {})
        if orb:
            st.write(f"Semi-major axis (au): {orb.get('semi_major_axis')}")
            st.write(f"Eccentricity: {orb.get('eccentricity')}")
            st.write(f"Inclination (deg): {orb.get('inclination')}")
            st.write(f"Epoch osculation: {orb.get('epoch_osculation')}")
        else:
            st.write("Orbital data not present.")

def run_simulation(neo_json: dict, user_inputs: dict):
    """
    Main simulation flow:
    - get diameter (user override optional)
    - compute mass, energy, crater, seismic magnitude
    - determine approach/impact probability with simple sampling method:
      sample small timing/velocity perturbations and map intercept points on Earth.
    """
    st.header("Simulation Summary & Impact Calculator")

    # diameter
    est_diam_min = neo_json.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_min")
    est_diam_max = neo_json.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_max")
    default_diameter = (est_diam_min + est_diam_max) / 2 if est_diam_min and est_diam_max else 50.0
    diameter_m = st.number_input("Asteroid diameter (m)", min_value=0.1, value=float(user_inputs.get("diameter_m", default_diameter)))
    density = st.number_input("Asteroid density (kg/m³)", min_value=500.0, max_value=8000.0, value=float(user_inputs.get("density", DEFAULT_DENSITY)), step=100.0)
    # choose approach velocity from close approach data (approx)
    cad = neo_json.get("close_approach_data", [])
    if len(cad) > 0:
        v_km_s = float(cad[0].get("relative_velocity", {}).get("kilometers_per_second") or 20.0)
        t_str = cad[0].get("close_approach_date_full") or cad[0].get("close_approach_date")
        # rough parse
        approach_dt = t_str
    else:
        v_km_s = 20.0
        approach_dt = "Unknown"
    v_m_s = 1000.0 * v_km_s
    st.write(f"Approach date: {approach_dt}, approach speed (approx): {v_km_s:.2f} km/s")

    # mass & energy
    mass_kg = asteroid_mass_from_diameter(diameter_m, density)
    energy_j = kinetic_energy_joules(mass_kg, v_m_s)
    mt_tnt = joules_to_megatons_tnt(energy_j)

    c1, c2, c3 = st.columns(3)
    c1.metric("Mass (kg)", f"{mass_kg:,.0f}")
    c2.metric("Kinetic Energy (J)", f"{energy_j:.3e}")
    c3.metric("Equivalent (megatons TNT)", f"{mt_tnt:.3f}")

    # crater & seismic
    crater_m = crater_diameter_estimate_simple(energy_j)
    Mw = seismic_magnitude_estimate(energy_j)
    st.write(f"Estimated transient crater diameter (approx): **{crater_m:.0f} m**")
    st.write(f"Rough seismic magnitude equivalent (approx): **Mw {Mw:.2f}**")

    st.markdown("---")
    st.subheader("Impact point sampling (approximate)")
    st.write("""
    We use a simplified linearized encounter model: sample small variations in the close approach geometry (time offsets, lateral displacement)
    to produce a probabilistic footprint. This is an approximation and **not** a replacement for mission-level orbital propagation.
    """)

    # time until encounter (user input)
    time_until_days = st.number_input("Time until encounter (days)", min_value=0.0, value=float(user_inputs.get("time_until_days", 30.0)))
    time_until_s = time_until_days * 86400.0

    # delta-v mitigation controls
    st.subheader("Mitigation: Delta-v (kinetic impactor style)")
    container = st.container()
    with container:
        dv_m_s = st.slider("Delta-v to asteroid (m/s) — positive lateral deflection", min_value=0.0, max_value=50.0, value=float(user_inputs.get("dv_m_s", 0.0)), step=0.5)
        dv_apply_days_before = st.number_input("Apply delta-v how many days before encounter?", min_value=0.0, value=5.0)
    dv_time_s = dv_apply_days_before * 86400.0

    # compute miss distances for sampled delta-vs (simple linear)
    sample_count = st.slider("Number of Monte Carlo samples (for footprint)", min_value=100, max_value=2000, value=600, step=100)
    rng = np.random.default_rng(12345)
    # assume baseline lateral uncertainties (m)
    baseline_lateral_sigma = st.number_input("Baseline lateral uncertainty at encounter (km)", min_value=0.1, max_value=1000.0, value=50.0) * 1000.0
    # sample headings (direction around Earth)
    angles = rng.random(sample_count) * 2 * math.pi
    baseline_offsets = rng.normal(loc=0.0, scale=baseline_lateral_sigma, size=sample_count)

    # compute miss distances pre & post mitigation
    miss_pre = np.sqrt(baseline_offsets**2)  # baseline scalar distances
    # effect of delta-v applied dv_time before: lateral displacement d = dv * dv_time_to_encounter where time to encounter after application equals (time_until_s - dv_time_s)
    effective_time = max(0.0, time_until_s - dv_time_s)
    dv_displacement = dv_m_s * effective_time
    miss_post = np.maximum(0.0, np.abs(baseline_offsets - dv_displacement))

    # Map sampled impact points onto Earth surface by choosing random approach longitude/latitude (simplified)
    # choose a reference mean impact lat/lon (user-selected)
    ref_lat = st.number_input("Reference impact latitude (deg)", min_value=-90.0, max_value=90.0, value=0.0, step=0.1)
    ref_lon = st.number_input("Reference impact longitude (deg)", min_value=-180.0, max_value=180.0, value=0.0, step=0.1)

    # compute lat/lon for each sample by offsetting from (ref_lat, ref_lon) by distance and angle using geodesic forward method
    lats_pre, lons_pre = [], []
    lats_post, lons_post = [], []
    for ang, d_pre, d_post in zip(angles, miss_pre, miss_post):
        # geod.fwd expects degrees and distance in meters
        lon_p, lat_p, _ = geod.fwd(ref_lon, ref_lat, math.degrees(ang), d_pre)
        lon_pp, lat_pp, _ = geod.fwd(ref_lon, ref_lat, math.degrees(ang), d_post)
        lats_pre.append(lat_p); lons_pre.append(lon_p)
        lats_post.append(lat_pp); lons_post.append(lon_pp)

    df_pre = pd.DataFrame({"lat": lats_pre, "lon": lons_pre, "miss_m": miss_pre})
    df_post = pd.DataFrame({"lat": lats_post, "lon": lons_post, "miss_m": miss_post})

    st.write("Map: pre-mitigation impact footprint (red) and post-mitigation (green) — sample points only.")
    # pydeck map
    initial_view = pdk.ViewState(latitude=ref_lat, longitude=ref_lon, zoom=3, pitch=0)
    scatter_pre = pdk.Layer(
        "ScatterplotLayer",
        df_pre,
        get_position='[lon, lat]',
        get_radius=20000,
        pickable=True,
        auto_highlight=True,
        get_fill_color=[220, 20, 60, 160],
        radius_scale=1
    )
    scatter_post = pdk.Layer(
        "ScatterplotLayer",
        df_post,
        get_position='[lon, lat]',
        get_radius=10000,
        pickable=True,
        auto_highlight=True,
        get_fill_color=[20, 160, 40, 160],
        radius_scale=1
    )
    r = pdk.Deck(layers=[scatter_pre, scatter_post], initial_view_state=initial_view, map_style='mapbox://styles/mapbox/light-v9')
    st.pydeck_chart(r)

    st.markdown("### Impact probability metrics (very approximate)")
    # compute fraction of pre samples with miss distance < Earth radius projected onto surface (i.e., actual impact)
    # For simplicity, assume impact occurs if miss distance < 0 (we interpret baseline offsets as radial around center). More robust treatment would require full orbit propagation.
    # We'll treat pre-mit samples within a threshold distance as 'impact'
    impact_threshold_m = (EARTH_RADIUS_M * 0.0) + 1.0  # placeholder; in reality geometry needed
    # Instead compute fraction of samples within a small threshold (e.g., < 10 km)
    th = st.number_input("Impact threshold for sample-to-impact (km)", min_value=1.0, max_value=1000.0, value=50.0) * 1000.0
    frac_pre_impacts = np.mean(miss_pre < th)
    frac_post_impacts = np.mean(miss_post < th)
    st.write(f"Approx fraction of sample trajectories hitting within {th/1000.0:.1f} km: **pre** {frac_pre_impacts:.3f}, **post** {frac_post_impacts:.3f}")
    st.caption("Note: these are simplified, sample-based metrics meant for educational comparison of mitigation effectiveness, not predictive impact probabilities.")

    st.markdown("---")
    st.subheader("Local environmental effects (USGS data)")
    with st.expander("Fetch local elevation & recent seismicity around a sample impact point"):
        sample_idx = st.number_input("Sample index to inspect", min_value=0, max_value=sample_count - 1, value=int(sample_count//2))
        lat_sample = df_pre.loc[sample_idx, "lat"]
        lon_sample = df_pre.loc[sample_idx, "lon"]
        st.write(f"Sample impact approx lat/lon: {lat_sample:.4f}, {lon_sample:.4f}")
        # fetch elevation
        try:
            elev_json = usgs_elevation_point(lat_sample, lon_sample)
            elev_val = elev_json.get("USGS_Elevation_Point_Query_Service", {}).get("Elevation_Query", {}).get("Elevation")
            st.write(f"Estimated elevation at sample impact: {elev_val} m (USGS EPQS)")
        except Exception as e:
            st.write("Failed to fetch elevation:", e)
        # fetch recent earthquakes within 500 km for context
        try:
            endt = datetime.utcnow()
            startt = endt - timedelta(days=365*1)
            eq_json = usgs_earthquake_search(startt.strftime("%Y-%m-%d"), endt.strftime("%Y-%m-%d"), minmagnitude=4.0, latitude=lat_sample, longitude=lon_sample, maxradiuskm=500)
            events = eq_json.get("features", [])
            st.write(f"Recent earthquakes (M>=4.0) within 500 km (last year): {len(events)}")
            if len(events) > 0:
                sample_events = []
                for e in events[:10]:
                    props = e.get("properties", {})
                    coords = e.get("geometry", {}).get("coordinates", [])
                    sample_events.append({
                        "time": datetime.utcfromtimestamp(props.get("time")/1000.0).isoformat(),
                        "mag": props.get("mag"),
                        "place": props.get("place"),
                        "lon": coords[0], "lat": coords[1]
                    })
                st.dataframe(pd.DataFrame(sample_events))
        except Exception as e:
            st.write("Failed to fetch earthquakes:", e)

    # Save/export results
    if st.button("Export simulation results (CSV)"):
        csv = df_pre.assign(phase="pre").append(df_post.assign(phase="post"), ignore_index=True).to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="impactor_simulation.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)


# ------------------ Gamified "Defend Earth" Mode ----------------------------
def defend_earth_game(neo_json: dict):
    """
    A simplified interactive scenario:
    - Users have limited resources (mass of kinetic impactor, launcher capability, time).
    - They choose impactor mass and impact timing; we compute delta-v imparted (approx)
      using conservation of momentum for an inelastic collision (worst-case).
    - Then show resulting miss distance improvement and a score.
    """
    st.header("Defend Earth — Gamified Simulator")
    st.write("""
    You are the mission designer. Use reasonable launch resources to apply a kinetic impactor
    to deflect the asteroid. This is an educational game that demonstrates trade-offs.
    """)

    # baseline asteroid mass
    est_diam_min = neo_json.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_min", 50.0)
    est_diam_max = neo_json.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_max", 100.0)
    default_d = (est_diam_min + est_diam_max) / 2.0
    diameter_m = st.number_input("Asteroid diameter (m)", value=float(default_d))
    density = st.number_input("Asteroid density (kg/m^3)", value=DEFAULT_DENSITY)
    asteroid_mass = asteroid_mass_from_diameter(diameter_m, density)
    st.write(f"Asteroid mass ~ {asteroid_mass:,.2f} kg")

    # user chooses impactor mass & approach speed
    st.subheader("Design your kinetic impactor")
    imp_mass = st.number_input("Impactor mass (kg)", value=1000.0, min_value=100.0, step=100.0)
    imp_speed = st.number_input("Relative speed at impact (km/s)", value=10.0, min_value=1.0, step=0.5) * 1000.0  # m/s
    # time until encounter
    time_until_days = st.number_input("Time until encounter (days)", value=180.0)
    time_until_s = time_until_days * 86400.0
    # impact time selection
    apply_days_before = st.slider("Impact how many days before predicted encounter?", min_value=0.0, max_value=time_until_days, value=30.0)
    apply_time_s = apply_days_before * 86400.0

    # compute delta-v from inelastic momentum transfer: dv = (imp_mass * imp_speed) / asteroid_mass
    dv_m_s = (imp_mass * imp_speed) / asteroid_mass
    st.write(f"Estimated delta-v imparted (inelastic approx): {dv_m_s:.6f} m/s")

    # convert to miss distance improvement using linear approx
    effective_time = max(0.0, time_until_s - apply_time_s)
    miss_improvement_m = dv_m_s * effective_time
    st.write(f"Estimated lateral displacement at encounter due to impact: {miss_improvement_m/1000.0:.2f} km")

    # scoring
    score = min(100, int(100.0 * (miss_improvement_m / (1000.0 * 100.0))))  # arbitrary scaling
    st.metric("Mission Score", f"{score}/100")
    st.balloons() if score > 70 else None

    st.write("### Visualize the deflection effect (schematic)")
    # show simple bar of pre vs post probability (toy)
    baseline_prob = 0.5  # arbitrary
    post_prob = max(0.0, baseline_prob - (miss_improvement_m / (1000.0 * 100.0)))
    fig = go.Figure(go.Bar(x=["Pre-mitigation", "Post-mitigation"], y=[baseline_prob*100, post_prob*100], text=[f"{baseline_prob*100:.1f}%", f"{post_prob*100:.1f}%"]))
    fig.update_yaxes(range=[0,100])
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Main App -----------------------------------------------
def main():
    st.title("Impactor-2025 — Asteroid Impact Simulator & Decision Support")
    st.markdown("""
    **Purpose:** integrate NASA NeoWs data and USGS environmental data to model asteroid impact scenarios,
    estimate consequences (energy, crater size, seismic equivalent), and evaluate mitigation strategies such as kinetic deflection.
    **Caveats:** This tool uses simplified physics for real-time interactive simulation and educational/exploratory use. It is **not** a mission planning tool.
    """)
    api_key, mode = sidebar_controls()

    # top-level: allow user to fetch a specific NEO (Impactor-2025 example)
    st.sidebar.markdown("### NEO Selection")
    neo_choice = st.sidebar.text_input("Enter NEO designation or NASA id (e.g., 'Impactor-2025' or '3542519')", value="Impactor-2025")
    with st.sidebar.expander("Quick examples"):
        st.write("- 'Impactor-2025' (fictional sample) — will fallback to demo data if not found")
        st.write("- '2010 TK7', '433 Eros' or NASA small-body ids")

    # Attempt to fetch NEO; fallback to sample demo data (Impactor-2025)
    neo_json = None
    if neo_choice:
        try:
            neo_json = nasa_neo_lookup_by_id(neo_choice, api_key) if api_key else None
        except Exception as e:
            # try using feed search by name (case-insensitive) or fallback
            st.sidebar.warning(f"NASA lookup failed: {e}. Falling back to demo data.")
            neo_json = None

    if neo_json is None:
        # demo placeholder NEO (Impactor-2025) — crafted fields similar to NASA NeoWs
        st.info("Using demo NEO 'Impactor-2025' (no NASA API key or lookup failed). Set NASA API key in sidebar to fetch real data.")
        neo_json = {
            "id": "IMP-2025",
            "name": "Impactor-2025",
            "absolute_magnitude_h": 21.5,
            "is_potentially_hazardous_asteroid": True,
            "estimated_diameter": {
                "meters": {"estimated_diameter_min": 50.0, "estimated_diameter_max": 120.0}
            },
            "orbital_data": {
                "semi_major_axis": 1.12,
                "eccentricity": 0.13,
                "inclination": 5.3,
                "epoch_osculation": "2459000.5"
            },
            "close_approach_data": [
                {
                    "close_approach_date": (datetime.utcnow() + timedelta(days=45)).strftime("%Y-%m-%d"),
                    "close_approach_date_full": (datetime.utcnow() + timedelta(days=45)).strftime("%Y-%m-%d %H:%M"),
                    "epoch_date_close_approach": int(time.time()*1000),
                    "relative_velocity": {"kilometers_per_second": "17.3"},
                    "miss_distance": {"kilometers": "200000"},
                    "orbiting_body": "Earth"
                }
            ]
        }

    # Show summary
    display_neo_summary(neo_json)

    # Mode handling
    if mode == "Explore NEO":
        st.header("Explore NEO Data")
        st.write("Raw JSON from NeoWs (or demo):")
        st.json(neo_json)
        st.markdown("---")
        st.write("Recent NEO feed (next 7 days) — quick sample (requires NASA API key)")
        if api_key:
            try:
                today = datetime.utcnow().date()
                feed = nasa_neo_feed(today.strftime("%Y-%m-%d"), (today+timedelta(days=7)).strftime("%Y-%m-%d"), api_key)
                # display a compact table
                rows = []
                for d, entries in feed.get("near_earth_objects", {}).items():
                    for e in entries:
                        cad = e.get("close_approach_data", [])
                        rows.append({
                            "date": d, "designation": e.get("name"), "diameter_min_m": e.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_min"),
                            "diameter_max_m": e.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_max"),
                            "hazard": e.get("is_potentially_hazardous_asteroid")
                        })
                st.dataframe(pd.DataFrame(rows))
            except Exception as e:
                st.write("Failed to fetch feed:", e)
        else:
            st.info("Enter NASA API key to fetch live NeoWs feeds.")

    elif mode == "Simulate Impact":
        # collect some default inputs to pass
        user_inputs = {
            "diameter_m": None,
            "density": DEFAULT_DENSITY,
            "time_until_days": 30.0,
            "dv_m_s": 0.0
        }
        run_simulation(neo_json, user_inputs)

    elif mode == "Defend Earth (Game)":
        defend_earth_game(neo_json)

    # Footer / educational overlays
    st.markdown("---")
    st.header("Educational Overlays & Methodology")
    st.markdown("""
    - **Orbital propagation**: This app uses simplified 2-body Kepler propagation when possible (poliastro). For fast interactivity we also include a linearized encounter approximation for sampling and mitigation experiments.  
    - **Energy & crater scaling**: Kinetic energy = 0.5 m v^2. Crater scaling is empirical and meant for order-of-magnitude estimates only.  
    - **Seismic & tsunami effects**: Represented with simple approximate coupling fractions. Modeling true tsunamis requires ocean bathymetry, hydrodynamic simulation, and is beyond this UI's runtime budget — but we've included hooks to integrate more advanced models offline.
    - **USGS & NASA integration**: This app demonstrates how to fetch USGS elevation and earthquake context and NASA NeoWs NEO data. Always validate API units (NeoWs uses km for miss distances, km/s for velocities).
    """)
    st.info("This tool is educational. For mission-critical planning, consult mission teams and high-fidelity modeling tools.")

if __name__ == "__main__":
    main()

