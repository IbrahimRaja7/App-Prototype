import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium, folium_static
import requests
import json
import math
from datetime import datetime, timedelta
import time
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import hashlib
import base64
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
st.set_page_config(
    page_title="🌍 Asteroid Impact Defense System",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CUSTOM CSS STYLING ==================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .stApp {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(90deg, #00ff88, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0,255,136,0.3);
    }
    
    .impact-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #ff0844, #ffb700, #00ff88, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,255,136,0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255,67,54,0.2), rgba(255,67,54,0.1));
        border-left: 4px solid #ff4336;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255,67,54,0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255,67,54,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,67,54,0); }
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,255,136,0.1));
        border-left: 4px solid #00ff88;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #00ff88, #00aaff);
        color: #0a0e27;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(0,255,136,0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0,255,136,0.6);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, rgba(15,12,41,0.9), rgba(36,36,62,0.9));
        backdrop-filter: blur(10px);
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: #00aaff;
        text-decoration: underline;
        text-decoration-style: dotted;
    }
    
    .defense-mode {
        background: linear-gradient(135deg, #ff0844, #ffb700);
        padding: 2px;
        border-radius: 15px;
    }
    
    .defense-mode-inner {
        background: #0a0e27;
        padding: 1.5rem;
        border-radius: 13px;
    }
    
    .progress-bar {
        width: 100%;
        height: 30px;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ff88, #00aaff);
        border-radius: 15px;
        transition: width 0.5s ease;
        box-shadow: 0 0 20px rgba(0,255,136,0.5);
    }
</style>
""", unsafe_allow_html=True)

# ================== SESSION STATE INITIALIZATION ==================
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'defense_score' not in st.session_state:
    st.session_state.defense_score = 0
if 'simulations_run' not in st.session_state:
    st.session_state.simulations_run = 0
if 'asteroids_deflected' not in st.session_state:
    st.session_state.asteroids_deflected = 0
if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = None
if 'neo_data' not in st.session_state:
    st.session_state.neo_data = None
if 'selected_asteroid' not in st.session_state:
    st.session_state.selected_asteroid = None
if 'mitigation_history' not in st.session_state:
    st.session_state.mitigation_history = []

# ================== PHYSICS CONSTANTS ==================
class PhysicsConstants:
    G = 6.67430e-11  # Gravitational constant (m³/kg/s²)
    EARTH_MASS = 5.972e24  # kg
    EARTH_RADIUS = 6.371e6  # meters
    AU = 1.496e11  # Astronomical Unit in meters
    EARTH_ORBITAL_VELOCITY = 29780  # m/s
    AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m³
    WATER_DENSITY = 1000  # kg/m³
    ROCK_DENSITY = 3000  # kg/m³ (typical asteroid density)
    ICE_DENSITY = 917  # kg/m³
    IRON_DENSITY = 7874  # kg/m³
    
    # Impact scaling constants
    CRATER_SCALING_FACTOR = 1.8  # Empirical crater scaling
    SEISMIC_EFFICIENCY = 0.0001  # Fraction of impact energy converted to seismic
    ATMOSPHERIC_ENTRY_VELOCITY = 11200  # m/s (minimum)
    
    # Mitigation parameters
    KINETIC_IMPACTOR_MASS = 500  # kg (typical spacecraft)
    NUCLEAR_YIELD_RANGE = (1e13, 1e15)  # Joules (1-100 MT)
    LASER_POWER_RANGE = (1e6, 1e9)  # Watts
    GRAVITY_TRACTOR_MASS = 10000  # kg

# ================== NASA NEO API INTERFACE ==================
class NasaNeoAPI:
    """Interface for NASA's Near-Earth Object Web Service"""
    
    BASE_URL = "https://api.nasa.gov/neo/rest/v1"
    API_KEY = "DEMO_KEY"  # In production, use a real API key
    
    @staticmethod
    def fetch_neo_feed(start_date: str = None, end_date: str = None) -> Optional[Dict]:
        """Fetch NEO data from NASA API"""
        try:
            if not start_date:
                start_date = datetime.now().strftime("%Y-%m-%d")
            if not end_date:
                end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            
            url = f"{NasaNeoAPI.BASE_URL}/feed"
            params = {
                "start_date": start_date,
                "end_date": end_date,
                "api_key": NasaNeoAPI.API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.warning(f"NASA API returned status code: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error fetching NEO data: {str(e)}")
            return None
    
    @staticmethod
    def fetch_asteroid_details(asteroid_id: str) -> Optional[Dict]:
        """Fetch detailed information about a specific asteroid"""
        try:
            url = f"{NasaNeoAPI.BASE_URL}/neo/{asteroid_id}"
            params = {"api_key": NasaNeoAPI.API_KEY}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error fetching asteroid details: {str(e)}")
            return None
    
    @staticmethod
    def parse_neo_data(neo_json: Dict) -> pd.DataFrame:
        """Parse NEO JSON data into structured DataFrame"""
        asteroids = []
        
        if neo_json and 'near_earth_objects' in neo_json:
            for date, objects in neo_json['near_earth_objects'].items():
                for obj in objects:
                    asteroid = {
                        'id': obj['id'],
                        'name': obj['name'],
                        'date': date,
                        'is_hazardous': obj['is_potentially_hazardous_asteroid'],
                        'diameter_min_m': obj['estimated_diameter']['meters']['estimated_diameter_min'],
                        'diameter_max_m': obj['estimated_diameter']['meters']['estimated_diameter_max'],
                        'diameter_avg_m': (obj['estimated_diameter']['meters']['estimated_diameter_min'] + 
                                         obj['estimated_diameter']['meters']['estimated_diameter_max']) / 2,
                        'absolute_magnitude': obj['absolute_magnitude_h'],
                        'velocity_kmps': float(obj['close_approach_data'][0]['relative_velocity']['kilometers_per_second']),
                        'velocity_ms': float(obj['close_approach_data'][0]['relative_velocity']['kilometers_per_second']) * 1000,
                        'miss_distance_km': float(obj['close_approach_data'][0]['miss_distance']['kilometers']),
                        'miss_distance_lunar': float(obj['close_approach_data'][0]['miss_distance']['lunar']),
                        'orbiting_body': obj['close_approach_data'][0]['orbiting_body']
                    }
                    
                    # Calculate additional properties
                    asteroid['volume_m3'] = (4/3) * math.pi * (asteroid['diameter_avg_m']/2)**3
                    asteroid['mass_kg'] = asteroid['volume_m3'] * PhysicsConstants.ROCK_DENSITY
                    asteroid['kinetic_energy_j'] = 0.5 * asteroid['mass_kg'] * asteroid['velocity_ms']**2
                    asteroid['kinetic_energy_mt'] = asteroid['kinetic_energy_j'] / 4.184e15  # Convert to megatons TNT
                    
                    asteroids.append(asteroid)
        
        return pd.DataFrame(asteroids)

# ================== USGS DATA INTERFACE ==================
class USGSDataInterface:
    """Interface for USGS earthquake and geological data"""
    
    EARTHQUAKE_API = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"
    
    @staticmethod
    def fetch_recent_earthquakes(min_magnitude: float = 2.5) -> Optional[pd.DataFrame]:
        """Fetch recent earthquake data from USGS"""
        try:
            url = f"{USGSDataInterface.EARTHQUAKE_API}/2.5_week.geojson"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                earthquakes = []
                
                for feature in data['features']:
                    props = feature['properties']
                    coords = feature['geometry']['coordinates']
                    
                    earthquakes.append({
                        'time': datetime.fromtimestamp(props['time']/1000),
                        'latitude': coords[1],
                        'longitude': coords[0],
                        'depth_km': coords[2],
                        'magnitude': props['mag'],
                        'place': props['place'],
                        'type': props['type']
                    })
                
                return pd.DataFrame(earthquakes)
            return None
        except Exception as e:
            st.warning(f"Could not fetch USGS earthquake data: {str(e)}")
            return None
    
    @staticmethod
    def get_elevation_data(lat: float, lon: float) -> float:
        """Get elevation at given coordinates (simplified - would use actual USGS DEM in production)"""
        # Simplified elevation model - in production, use USGS National Map API
        base_elevation = 0
        
        # Simulate mountainous regions
        if 25 <= abs(lat) <= 45:
            base_elevation += np.random.uniform(0, 2000)
        
        # Simulate oceanic regions
        if abs(lat) > 60 or (abs(lon) > 150 and abs(lon) < 180):
            base_elevation = -np.random.uniform(100, 4000)
        
        return base_elevation
    
    @staticmethod
    def get_population_density(lat: float, lon: float) -> float:
        """Estimate population density at location (simplified model)"""
        # Major population centers (simplified)
        cities = [
            (40.7128, -74.0060, 27000),  # New York City
            (35.6762, 139.6503, 16000),  # Tokyo
            (51.5074, -0.1278, 5700),    # London
            (48.8566, 2.3522, 21000),     # Paris
            (37.7749, -122.4194, 7200),  # San Francisco
            (-23.5505, -46.6333, 7400),  # São Paulo
            (19.4326, -99.1332, 9200),   # Mexico City
            (28.6139, 77.2090, 11300),   # Delhi
            (31.2304, 121.4737, 27000),  # Shanghai
            (34.0522, -118.2437, 8500)   # Los Angeles
        ]
        
        max_density = 0
        for city_lat, city_lon, density in cities:
            distance = math.sqrt((lat - city_lat)**2 + (lon - city_lon)**2)
            if distance < 5:  # Within ~500km
                city_density = density * math.exp(-distance/2)
                max_density = max(max_density, city_density)
        
        # Background rural density
        if max_density == 0:
            max_density = np.random.uniform(10, 100)
        
        return max_density

# ================== ORBITAL MECHANICS ==================
class OrbitalMechanics:
    """Advanced orbital mechanics calculations"""
    
    @staticmethod
    def kepler_to_cartesian(a: float, e: float, i: float, omega: float, 
                          Omega: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Keplerian orbital elements to Cartesian coordinates"""
        # Semi-latus rectum
        p = a * (1 - e**2)
        r_mag = p / (1 + e * math.cos(nu))
        
        # Position in orbital plane
        r_orbital = np.array([
            r_mag * math.cos(nu),
            r_mag * math.sin(nu),
            0
        ])
        
        # Velocity in orbital plane
        mu = PhysicsConstants.G * PhysicsConstants.EARTH_MASS
        h = math.sqrt(mu * p)
        
        v_orbital = np.array([
            -mu/h * math.sin(nu),
            mu/h * (e + math.cos(nu)),
            0
        ])
        
        # Rotation matrices
        cos_omega, sin_omega = math.cos(omega), math.sin(omega)
        cos_Omega, sin_Omega = math.cos(Omega), math.sin(Omega)
        cos_i, sin_i = math.cos(i), math.sin(i)
        
        # Transform to inertial frame
        R = np.array([
            [cos_Omega*cos_omega - sin_Omega*sin_omega*cos_i,
             -cos_Omega*sin_omega - sin_Omega*cos_omega*cos_i,
             sin_Omega*sin_i],
            [sin_Omega*cos_omega + cos_Omega*sin_omega*cos_i,
             -sin_Omega*sin_omega + cos_Omega*cos_omega*cos_i,
             -cos_Omega*sin_i],
            [sin_omega*sin_i, cos_omega*sin_i, cos_i]
        ])
        
        r = R @ r_orbital
        v = R @ v_orbital
        
        return r, v
    
    @staticmethod
    def propagate_orbit(r0: np.ndarray, v0: np.ndarray, dt: float, 
                       n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate orbit using numerical integration"""
        def dynamics(state, t):
            r = state[:3]
            v = state[3:6]
            
            r_mag = np.linalg.norm(r)
            mu = PhysicsConstants.G * PhysicsConstants.EARTH_MASS
            
            a = -mu * r / r_mag**3
            
            return np.concatenate([v, a])
        
        initial_state = np.concatenate([r0, v0])
        t_span = np.linspace(0, dt, n_steps)
        
        solution = odeint(dynamics, initial_state, t_span)
        
        positions = solution[:, :3]
        velocities = solution[:, 3:6]
        
        return positions, velocities
    
    @staticmethod
    def calculate_close_approach(asteroid_orbit: Dict, earth_position: np.ndarray) -> Dict:
        """Calculate closest approach parameters"""
        # Simplified close approach calculation
        min_distance = asteroid_orbit.get('miss_distance_km', 1e6) * 1000  # Convert to meters
        relative_velocity = asteroid_orbit.get('velocity_ms', 20000)
        
        # Impact parameter
        b = min_distance
        
        # Gravitational focusing
        v_inf = relative_velocity
        v_esc = math.sqrt(2 * PhysicsConstants.G * PhysicsConstants.EARTH_MASS / PhysicsConstants.EARTH_RADIUS)
        
        # Enhanced velocity due to Earth's gravity
        v_impact = math.sqrt(v_inf**2 + v_esc**2)
        
        # Deflection angle
        theta = 2 * math.atan(PhysicsConstants.G * PhysicsConstants.EARTH_MASS / (b * v_inf**2))
        
        return {
            'min_distance': min_distance,
            'relative_velocity': relative_velocity,
            'impact_parameter': b,
            'impact_velocity': v_impact,
            'deflection_angle': math.degrees(theta),
            'gravitational_focusing': v_impact / v_inf
        }

# ================== IMPACT PHYSICS ==================
class ImpactPhysics:
    """Comprehensive impact physics modeling"""
    
    @staticmethod
    def calculate_impact_energy(mass: float, velocity: float) -> Dict:
        """Calculate various energy metrics for impact"""
        kinetic_energy = 0.5 * mass * velocity**2
        
        # Energy comparisons
        hiroshima_yield = 6.3e13  # Joules (15 kilotons)
        megaton_tnt = 4.184e15    # Joules
        
        return {
            'kinetic_energy_j': kinetic_energy,
            'kinetic_energy_mt': kinetic_energy / megaton_tnt,
            'hiroshima_equivalents': kinetic_energy / hiroshima_yield,
            'earthquake_magnitude': ImpactPhysics.energy_to_magnitude(kinetic_energy),
            'energy_density': kinetic_energy / (4/3 * math.pi * PhysicsConstants.EARTH_RADIUS**3)
        }
    
    @staticmethod
    def energy_to_magnitude(energy: float) -> float:
        """Convert energy to Richter magnitude equivalent"""
        # Gutenberg-Richter relation: log10(E) = 4.8 + 1.5*M
        if energy <= 0:
            return 0
        return (math.log10(energy) - 4.8) / 1.5
    
    @staticmethod
    def calculate_crater_dimensions(energy: float, impact_angle: float = 45, 
                                   target_density: float = 2500) -> Dict:
        """Calculate crater dimensions using scaling laws"""
        # Holsapple-Schmidt scaling laws
        K1 = 0.14  # Scaling constant
        nu = 0.4   # Scaling exponent
        
        # Effective energy (accounting for impact angle)
        E_eff = energy * math.sin(math.radians(impact_angle))
        
        # Transient crater diameter
        D_t = K1 * (E_eff / PhysicsConstants.G)**(nu)
        
        # Final crater diameter (accounting for collapse)
        if D_t > 3000:  # Complex crater
            D_f = D_t * 1.18
            depth = D_f / 15  # Depth-to-diameter ratio for complex craters
        else:  # Simple crater
            D_f = D_t * 1.25
            depth = D_f / 5   # Depth-to-diameter ratio for simple craters
        
        # Ejecta blanket
        ejecta_radius = D_f * 2.5
        ejecta_volume = math.pi * depth * (ejecta_radius**2 - (D_f/2)**2)
        
        return {
            'transient_diameter': D_t,
            'final_diameter': D_f,
            'depth': depth,
            'rim_height': depth * 0.04,
            'ejecta_radius': ejecta_radius,
            'ejecta_volume': ejecta_volume,
            'excavated_mass': ejecta_volume * target_density,
            'crater_type': 'Complex' if D_t > 3000 else 'Simple'
        }
    
    @staticmethod
    def atmospheric_entry_effects(mass: float, velocity: float, diameter: float, 
                                 entry_angle: float = 45) -> Dict:
        """Model atmospheric entry and fragmentation"""
        # Initial parameters
        area = math.pi * (diameter/2)**2
        
        # Atmospheric density profile (exponential)
        scale_height = 8500  # meters
        
        # Drag coefficient
        Cd = 0.92  # Sphere
        
        # Peak deceleration altitude
        rho_s = PhysicsConstants.ROCK_DENSITY
        H = scale_height
        theta = math.radians(entry_angle)
        
        # Fragmentation pressure threshold
        strength = 1e7  # Pa (typical for stony asteroids)
        
        # Calculate fragmentation altitude
        z_frag = -H * math.log(strength / (0.5 * PhysicsConstants.AIR_DENSITY_SEA_LEVEL * velocity**2))
        
        if z_frag < 0:
            z_frag = 0  # Fragmentation at ground level
        
        # Energy deposition in atmosphere
        energy_deposited = 0.5 * mass * velocity**2 * (1 - math.exp(-z_frag/H))
        
        # Airblast overpressure at ground
        if z_frag > 0:
            # Sedov-Taylor blast wave
            E_blast = energy_deposited * 0.15  # Fraction converted to blast
            overpressure = 1.4e5 * (E_blast / 4.184e15)**(1/3) / (z_frag/1000)
        else:
            overpressure = 0
        
        return {
            'fragmentation_altitude': z_frag,
            'energy_deposited_atmosphere': energy_deposited,
            'peak_brightness_magnitude': -2.5 * math.log10(energy_deposited/1e12),
            'airblast_overpressure': overpressure,
            'thermal_radiation': energy_deposited * 0.3,
            'survival_fraction': math.exp(-z_frag/(2*H)) if z_frag > 0 else 1.0
        }
    
    @staticmethod
    def tsunami_modeling(impact_energy: float, impact_location: Tuple[float, float], 
                        water_depth: float) -> Dict:
        """Model tsunami generation and propagation"""
        if water_depth <= 0:
            return {
                'tsunami_generated': False,
                'initial_wave_height': 0,
                'wavelength': 0,
                'propagation_speed': 0,
                'coastal_runup': 0
            }
        
        # Ward-Asphaug model for asteroid-generated tsunamis
        # Initial cavity parameters
        R_cavity = 0.7 * (impact_energy / 1e20)**(0.25) * 1000  # meters
        
        # Initial wave amplitude
        H_0 = min(R_cavity * 0.06, water_depth * 0.8)
        
        # Wavelength
        wavelength = 2 * math.pi * R_cavity
        
        # Propagation speed (shallow water approximation)
        c = math.sqrt(PhysicsConstants.G * water_depth)
        
        # Dispersion relation for deep water
        if water_depth > wavelength/2:
            k = 2 * math.pi / wavelength
            c = math.sqrt(PhysicsConstants.G / k)
        
        # Coastal amplification (simplified)
        runup_factor = 4  # Typical amplification
        coastal_runup = H_0 * runup_factor
        
        # Attenuation with distance (geometric spreading)
        distances = np.array([100, 500, 1000, 5000, 10000])  # km
        wave_heights = H_0 * np.sqrt(R_cavity / (distances * 1000))
        
        return {
            'tsunami_generated': True,
            'initial_wave_height': H_0,
            'cavity_radius': R_cavity,
            'wavelength': wavelength,
            'propagation_speed': c,
            'coastal_runup': coastal_runup,
            'period': wavelength / c,
            'wave_heights_vs_distance': {
                'distances_km': distances.tolist(),
                'wave_heights_m': wave_heights.tolist()
            }
        }
    
    @staticmethod
    def seismic_effects(impact_energy: float, distance_km: float) -> Dict:
        """Calculate seismic effects from impact"""
        # Seismic efficiency (fraction of energy converted to seismic waves)
        eta = PhysicsConstants.SEISMIC_EFFICIENCY
        E_seismic = impact_energy * eta
        
        # Moment magnitude
        M_w = (2/3) * (math.log10(E_seismic) - 4.8)
        
        # Peak ground acceleration (PGA) using attenuation relation
        # Boore-Atkinson GMPE
        if distance_km > 0:
            log_pga = M_w - 3.512 - 0.006 * distance_km - math.log10(distance_km)
            pga = 10**log_pga * 9.81
