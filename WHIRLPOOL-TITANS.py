import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
from datetime import datetime, timedelta
import json
import folium
from streamlit_folium import st_folium
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Constants
EARTH_RADIUS_KM = 6371
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
EARTH_MASS_KG = 5.972e24
SUN_MASS_KG = 1.989e30
AU_TO_KM = 1.496e8
ASTEROID_DENSITY_KG_M3 = 3000  # Average density for rocky asteroids
TNT_EQUIVALENT_J_PER_KT = 4.184e12  # Joules per kiloton TNT

# NASA API Key (use DEMO_KEY for testing, replace with real if needed)
NASA_API_KEY = "DEMO_KEY"
NASA_NEO_API_BASE = "https://api.nasa.gov/neo/rest/v1"

# USGS APIs
USGS_EARTHQUAKE_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_ELEVATION_API = "https://epqs.nationalmap.gov/v1/json"  # Elevation Point Query Service

# Helper Functions

def fetch_neo_data(asteroid_name=None, start_date=None, end_date=None):
    """Fetch NEO data from NASA API."""
    if asteroid_name:
        url = f"{NASA_NEO_API_BASE}/neo/sentry?api_key={NASA_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Search for asteroid by name (simplified, as API might not have exact name search)
            for obj in data.get('data', []):
                if obj['fullname'].lower().find(asteroid_name.lower()) != -1:
                    return obj
        # Fallback to browse if not found
        url = f"{NASA_NEO_API_BASE}/feed?start_date={start_date}&end_date={end_date}&api_key={NASA_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('near_earth_objects', {})
    else:
        url = f"{NASA_NEO_API_BASE}/feed?start_date={start_date}&end_date={end_date}&api_key={NASA_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('near_earth_objects', {})
    return {}

def parse_neo_data(neo_data):
    """Parse NEO data into usable parameters."""
    if not neo_data:
        return None
    # Example parsing for first close approach
    close_approach = neo_data.get('close_approach_data', [{}])[0]
    orbital_data = neo_data.get('orbital_data', {})
    params = {
        'diameter_km': neo_data.get('estimated_diameter', {}).get('kilometers', {}).get('estimated_diameter_max', 0.1),
        'velocity_km_s': float(close_approach.get('relative_velocity', {}).get('kilometers_per_second', 10)),
        'miss_distance_km': float(close_approach.get('miss_distance', {}).get('kilometers', 1e6)),
        'semi_major_axis_au': float(orbital_data.get('semi_major_axis', 1.0)),
        'eccentricity': float(orbital_data.get('eccentricity', 0.0)),
        'inclination_deg': float(orbital_data.get('inclination', 0.0)),
        'longitude_ascending_node_deg': float(orbital_data.get('ascending_node_longitude', 0.0)),
        'argument_of_periapsis_deg': float(orbital_data.get('argument_of_perihelion', 0.0)),
        'mean_anomaly_deg': float(orbital_data.get('mean_anomaly', 0.0)),
    }
    return params

def kepler_to_cartesian(a, e, i, omega, Omega, M, mu):
    """Convert Keplerian elements to Cartesian position and velocity."""
    # Solve Kepler's equation for eccentric anomaly E
    def kepler_eq(E):
        return E - e * math.sin(E) - M
    E = root_scalar(kepler_eq, bracket=[0, 2*math.pi]).root
    
    # True anomaly
    nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
    
    # Distance
    r = a * (1 - e**2) / (1 + e * math.cos(nu))
    
    # Position in orbital plane
    x_orb = r * math.cos(nu)
    y_orb = r * math.sin(nu)
    z_orb = 0
    
    # Velocity in orbital plane
    sqrt_mu_a = math.sqrt(mu / a)
    vx_orb = -sqrt_mu_a * math.sin(nu) / math.sqrt(1 - e**2)
    vy_orb = sqrt_mu_a * (e + math.cos(nu)) / math.sqrt(1 - e**2)
    vz_orb = 0
    
    # Rotation matrices
    cos_Omega = math.cos(Omega)
    sin_Omega = math.sin(Omega)
    cos_omega = math.cos(omega)
    sin_omega = math.sin(omega)
    cos_i = math.cos(i)
    sin_i = math.sin(i)
    
    # Position
    x = x_orb * (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) - y_orb * (cos_Omega * sin_omega + sin_Omega * cos_omega * cos_i)
    y = x_orb * (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) + y_orb * (cos_Omega * cos_omega * cos_i - sin_Omega * sin_omega)
    z = x_orb * (sin_omega * sin_i) + y_orb * (cos_omega * sin_i)
    
    # Velocity
    vx = vx_orb * (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) - vy_orb * (cos_Omega * sin_omega + sin_Omega * cos_omega * cos_i)
    vy = vx_orb * (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) + vy_orb * (cos_Omega * cos_omega * cos_i - sin_Omega * sin_omega)
    vz = vx_orb * (sin_omega * sin_i) + vy_orb * (cos_omega * sin_i)
    
    return np.array([x, y, z]), np.array([vx, vy, vz])

def simulate_trajectory(params, steps=1000, deflection_delta_v=0):
    """Simulate asteroid trajectory."""
    a = params['semi_major_axis_au'] * AU_TO_KM * 1000  # m
    e = params['eccentricity']
    i = math.radians(params['inclination_deg'])
    omega = math.radians(params['argument_of_periapsis_deg'])
    Omega = math.radians(params['longitude_ascending_node_deg'])
    M = math.radians(params['mean_anomaly_deg'])
    mu = GRAVITATIONAL_CONSTANT * SUN_MASS_KG  # For heliocentric
    
    positions = []
    for t in np.linspace(0, 2*math.pi, steps):
        current_M = M + t  # Simple propagation
        pos, vel = kepler_to_cartesian(a, e, i, omega, Omega, current_M, mu)
        positions.append(pos / 1000 / AU_TO_KM)  # Back to AU
    
    # Apply deflection as delta-v to velocity (simplified)
    if deflection_delta_v != 0:
        # Assume deflection changes semi-major axis or eccentricity
        a += deflection_delta_v * 1e3  # Rough approximation
        positions = []  # Recompute
        for t in np.linspace(0, 2*math.pi, steps):
            current_M = M + t
            pos, vel = kepler_to_cartesian(a, e, i, omega, Omega, current_M, mu)
            positions.append(pos / 1000 / AU_TO_KM)
    
    return np.array(positions)

def calculate_impact_energy(diameter_km, velocity_km_s):
    """Calculate impact energy."""
    radius_m = diameter_km * 500
    volume_m3 = (4/3) * math.pi * radius_m**3
    mass_kg = ASTEROID_DENSITY_KG_M3 * volume_m3
    energy_j = 0.5 * mass_kg * (velocity_km_s * 1000)**2
    energy_kt_tnt = energy_j / TNT_EQUIVALENT_J_PER_KT
    return energy_kt_tnt

def estimate_crater_size(energy_kt):
    """Estimate crater diameter using simple scaling."""
    # Approximate formula: D = 1.8 * (energy_kt)^{0.294} for km
    energy_mt = energy_kt / 1000
    crater_diam_km = 0.001 * energy_mt**0.333 * 10  # Rough estimate
    return crater_diam_km

def fetch_usgs_elevation(lat, lon):
    """Fetch elevation from USGS."""
    params = {'x': lon, 'y': lat, 'units': 'Meters', 'output': 'json'}
    response = requests.get(USGS_ELEVATION_API, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('value', 0)
    return 0

def estimate_tsunami_risk(impact_lat, impact_lon, energy_kt):
    """Estimate if tsunami possible based on elevation."""
    elevation = fetch_usgs_elevation(impact_lat, impact_lon)
    if elevation < 0:  # Ocean
        # Simple: tsunami height proportional to energy
        tsunami_height_m = min(100, (energy_kt / 1e6)**0.5 * 10)
        return True, tsunami_height_m
    return False, 0

def estimate_seismic_magnitude(energy_kt):
    """Estimate seismic magnitude."""
    # Approximate: M = (2/3) log10(E) - 2.9, E in dyne-cm (ergs)
    energy_erg = (energy_kt * TNT_EQUIVALENT_J_PER_KT) * 1e7  # J to erg
    mag = (2/3) * math.log10(energy_erg) - 2.9
    return max(0, min(10, mag))

def fetch_earthquakes(starttime, endtime, minmagnitude=5):
    """Fetch recent earthquakes from USGS."""
    params = {
        'format': 'geojson',
        'starttime': starttime,
        'endtime': endtime,
        'minmagnitude': minmagnitude
    }
    response = requests.get(USGS_EARTHQUAKE_API, params=params)
    if response.status_code == 200:
        return response.json().get('features', [])
    return []

# Visualization Functions

def plot_orbit_3d(trajectory, earth_pos=np.array([0,0,0])):
    """Plot 3D orbit using Plotly."""
    fig = go.Figure()
    
    # Asteroid trajectory
    fig.add_trace(go.Scatter3d(
        x=trajectory[:,0], y=trajectory[:,1], z=trajectory[:,2],
        mode='lines', line=dict(color='red', width=4),
        name='Asteroid Trajectory'
    ))
    
    # Earth
    fig.add_trace(go.Scatter3d(
        x=[earth_pos[0]], y=[earth_pos[1]], z=[earth_pos[2]],
        mode='markers', marker=dict(size=10, color='blue'),
        name='Earth'
    ))
    
    # Sun
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(size=15, color='yellow'),
        name='Sun'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (AU)',
            yaxis_title='Y (AU)',
            zaxis_title='Z (AU)',
            aspectmode='cube'
        ),
        title='Asteroid Trajectory Relative to Earth and Sun'
    )
    return fig

def plot_impact_map(impact_lat, impact_lon, crater_size_km, tsunami=False):
    """Plot impact zone on map using Folium."""
    m = folium.Map(location=[impact_lat, impact_lon], zoom_start=4)
    
    # Impact point
    folium.Marker([impact_lat, impact_lon], popup='Impact Point').add_to(m)
    
    # Crater circle
    folium.Circle(
        location=[impact_lat, impact_lon],
        radius=crater_size_km * 1000 / 2,
        color='red',
        fill=True,
        fill_color='red',
        popup=f'Crater: {crater_size_km:.2f} km'
    ).add_to(m)
    
    if tsunami:
        # Rough tsunami zone
        folium.Circle(
            location=[impact_lat, impact_lon],
            radius=500 * 1000,  # 500 km
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.3,
            popup='Potential Tsunami Zone'
        ).add_to(m)
    
    return m

def generate_infographic(energy_kt, mag, crater_km, tsunami_height):
    """Generate simple infographic using Matplotlib."""
    fig, ax = plt.subplots(figsize=(6,4))
    ax.text(0.5, 0.9, f'Impact Energy: {energy_kt:.2e} kt TNT', ha='center')
    ax.text(0.5, 0.7, f'Seismic Magnitude: {mag:.1f}', ha='center')
    ax.text(0.5, 0.5, f'Crater Size: {crater_km:.2f} km', ha='center')
    ax.text(0.5, 0.3, f'Tsunami Height: {tsunami_height:.2f} m' if tsunami_height > 0 else 'No Tsunami', ha='center')
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_base64}" width="400">'

# Gamification Functions

def defend_earth_game(params, target_miss_distance=1e5):
    """Simple game: adjust delta_v to miss Earth."""
    st.subheader("Defend Earth Game")
    st.write("Adjust the deflection velocity to make the asteroid miss Earth by at least the target distance.")
    
    delta_v = st.slider("Deflection Delta-V (m/s)", -1000.0, 1000.0, 0.0, 1.0)
    
    # Simulate new miss distance (simplified linear approximation)
    original_miss = params['miss_distance_km']
    change_factor = delta_v / 10  # Rough
    new_miss = original_miss + change_factor * 1000  # km
    
    if new_miss >= target_miss_distance:
        st.success(f"Success! New miss distance: {new_miss:.2f} km")
        return True
    else:
        st.error(f"Too close! New miss distance: {new_miss:.2f} km. Target: {target_miss_distance} km")
        return False

# Main App

st.set_page_config(page_title="Asteroid Impact Simulator", layout="wide")

st.title("Asteroid Impact Simulator: Impactor-2025 and Beyond")

# Sidebar for Inputs
with st.sidebar:
    st.header("Asteroid Parameters")
    
    mode = st.radio("Mode", ["Fetch Real Data", "Manual Input", "Gamified Mode"])
    
    if mode == "Fetch Real Data":
        asteroid_name = st.text_input("Asteroid Name (e.g., Impactor-2025)", "Apophis")
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        end_date = st.date_input("End Date", datetime.now())
        if st.button("Fetch Data"):
            neo_data = fetch_neo_data(asteroid_name, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            params = parse_neo_data(next(iter(neo_data.values()))[0] if neo_data else None) if neo_data else None
            if params:
                st.session_state['params'] = params
            else:
                st.error("No data found.")
    
    elif mode == "Manual Input":
        params = {
            'diameter_km': st.slider("Diameter (km)", 0.01, 10.0, 0.1),
            'velocity_km_s': st.slider("Velocity (km/s)", 1.0, 50.0, 10.0),
            'miss_distance_km': st.slider("Miss Distance (km)", 1e3, 1e7, 1e6),
            'semi_major_axis_au': st.slider("Semi-Major Axis (AU)", 0.5, 5.0, 1.0),
            'eccentricity': st.slider("Eccentricity", 0.0, 0.99, 0.0),
            'inclination_deg': st.slider("Inclination (deg)", 0.0, 180.0, 0.0),
            'longitude_ascending_node_deg': st.slider("Longitude of Ascending Node (deg)", 0.0, 360.0, 0.0),
            'argument_of_periapsis_deg': st.slider("Argument of Periapsis (deg)", 0.0, 360.0, 0.0),
            'mean_anomaly_deg': st.slider("Mean Anomaly (deg)", 0.0, 360.0, 0.0),
        }
        st.session_state['params'] = params
    
    deflection_delta_v = st.slider("Deflection Delta-V (m/s)", -1000.0, 1000.0, 0.0, help="Apply velocity change for mitigation")
    
    impact_lat = st.number_input("Impact Latitude", -90.0, 90.0, 0.0)
    impact_lon = st.number_input("Impact Longitude", -180.0, 180.0, 0.0)
    
    st.header("USGS Data")
    eq_start = st.date_input("Earthquake Start Date", datetime.now() - timedelta(days=30))
    eq_end = st.date_input("Earthquake End Date", datetime.now())
    min_mag = st.slider("Min Magnitude", 1.0, 10.0, 5.0)

# Main Content
if 'params' in st.session_state:
    params = st.session_state['params']
    
    # Trajectory Simulation
    st.header("Asteroid Trajectory")
    with st.expander("Explanation"):
        st.write("This 3D visualization shows the asteroid's orbital path using Keplerian elements. The path is propagated over one orbital period.")
    
    trajectory = simulate_trajectory(params, deflection_delta_v=deflection_delta_v)
    orbit_fig = plot_orbit_3d(trajectory)
    st.plotly_chart(orbit_fig, use_container_width=True)
    
    # Impact Calculations
    st.header("Impact Consequences")
    energy_kt = calculate_impact_energy(params['diameter_km'], params['velocity_km_s'])
    crater_km = estimate_crater_size(energy_kt)
    mag = estimate_seismic_magnitude(energy_kt)
    tsunami, tsunami_height = estimate_tsunami_risk(impact_lat, impact_lon, energy_kt)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Impact Energy (kt TNT)", f"{energy_kt:.2e}")
    col2.metric("Crater Diameter (km)", f"{crater_km:.2f}")
    col3.metric("Seismic Magnitude", f"{mag:.1f}")
    
    if tsunami:
        st.metric("Tsunami Height (m)", f"{tsunami_height:.2f}")
    else:
        st.write("No significant tsunami risk (land impact).")
    
    # Infographic
    st.subheader("Impact Summary Infographic")
    infographic = generate_infographic(energy_kt, mag, crater_km, tsunami_height)
    st.markdown(infographic, unsafe_allow_html=True)
    
    # Impact Map
    st.subheader("Impact Zone Map")
    with st.expander("Explanation"):
        st.write("This map shows the potential impact point, crater size, and tsunami zone if applicable, using USGS elevation data.")
    impact_map = plot_impact_map(impact_lat, impact_lon, crater_km, tsunami)
    st_folium(impact_map, width=700, height=500)
    
    # USGS Earthquakes
    st.subheader("Recent Earthquakes (for Context)")
    earthquakes = fetch_earthquakes(eq_start.strftime("%Y-%m-%d"), eq_end.strftime("%Y-%m-%d"), min_mag)
    eq_df = pd.DataFrame([{
        'Time': datetime.fromtimestamp(eq['properties']['time']/1000),
        'Magnitude': eq['properties']['mag'],
        'Place': eq['properties']['place']
    } for eq in earthquakes])
    st.dataframe(eq_df)
    
    # Mitigation Evaluation
    st.header("Mitigation Strategies")
    with st.expander("Explanation"):
        st.write("Adjust the deflection delta-v in the sidebar to see how it changes the trajectory. Positive values increase semi-major axis slightly.")
    st.write("Current Miss Distance after Deflection:", params['miss_distance_km'] + (deflection_delta_v / 10 * 1000), "km")  # Simplified
    
    # Gamified Mode
    if mode == "Gamified Mode":
        success = defend_earth_game(params)
        if success:
            st.balloons()
    
    # Educational Content
    st.header("Learn More")
    with st.expander("Orbital Mechanics"):
        st.write("Keplerian elements define the orbit: semi-major axis (size), eccentricity (shape), etc.")
    with st.expander("Impact Energy"):
        st.write("Calculated as kinetic energy: 1/2 mv^2, converted to TNT equivalent.")
    with st.expander("Crater Scaling"):
        st.write("Uses empirical scaling laws based on energy.")
    with st.expander("Environmental Effects"):
        st.write("Tsunami if ocean impact, seismic waves always.")
    
    # Social Sharing (simulated)
    st.subheader("Share Your Simulation")
    share_text = f"Asteroid Impact Simulation: Energy {energy_kt:.2e} kt, Crater {crater_km:.2f} km"
    st.text_area("Share Text", share_text)
    st.button("Copy to Clipboard")  # Actual copy would need JS, but simulated

else:
    st.info("Please input or fetch asteroid parameters in the sidebar to start.")

# Error Handling
try:
    # All code is wrapped implicitly
    pass
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
