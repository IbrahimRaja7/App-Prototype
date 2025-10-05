import streamlit as st
import numpy as np
import pandas as pd


from datetime import datetime, timedelta
import requests
import json
from scipy.integrate import odeint
import math

# Page configuration
st.set_page_config(
    page_title="Impactor-2025 Defense System",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .stAlert {
        background-color: rgba(255, 107, 107, 0.1);
        border-left: 5px solid #FF6B6B;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .defense-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'defense_score' not in st.session_state:
    st.session_state.defense_score = 0
if 'scenarios_completed' not in st.session_state:
    st.session_state.scenarios_completed = 0

# Constants
AU = 1.496e11  # Astronomical Unit in meters
EARTH_RADIUS = 6.371e6  # meters
G = 6.67430e-11  # Gravitational constant
EARTH_MASS = 5.972e24  # kg
TNT_JOULE = 4.184e9  # Joules per kiloton of TNT

class AsteroidPhysics:
    """Advanced asteroid physics calculations"""
    
    @staticmethod
    def calculate_orbital_elements(a, e, i, omega, w, M, t):
        """Calculate position using Keplerian orbital elements"""
        # Convert degrees to radians
        i_rad = np.radians(i)
        omega_rad = np.radians(omega)
        w_rad = np.radians(w)
        M_rad = np.radians(M)
        
        # Solve Kepler's equation for Eccentric Anomaly
        E = M_rad
        for _ in range(10):
            E = M_rad + e * np.sin(E)
        
        # True anomaly
        nu = 2 * np.arctan2(
            np.sqrt(1 + e) * np.sin(E / 2),
            np.sqrt(1 - e) * np.cos(E / 2)
        )
        
        # Distance from Sun
        r = a * (1 - e * np.cos(E))
        
        # Position in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        
        # Rotate to ecliptic coordinates
        x = (np.cos(omega_rad) * np.cos(w_rad) - np.sin(omega_rad) * np.sin(w_rad) * np.cos(i_rad)) * x_orb + \
            (-np.cos(omega_rad) * np.sin(w_rad) - np.sin(omega_rad) * np.cos(w_rad) * np.cos(i_rad)) * y_orb
        
        y = (np.sin(omega_rad) * np.cos(w_rad) + np.cos(omega_rad) * np.sin(w_rad) * np.cos(i_rad)) * x_orb + \
            (-np.sin(omega_rad) * np.sin(w_rad) + np.cos(omega_rad) * np.cos(w_rad) * np.cos(i_rad)) * y_orb
        
        z = (np.sin(w_rad) * np.sin(i_rad)) * x_orb + (np.cos(w_rad) * np.sin(i_rad)) * y_orb
        
        return x, y, z
    
    @staticmethod
    def calculate_impact_energy(diameter, velocity, density=3000):
        """Calculate impact energy in Joules and TNT equivalent"""
        radius = diameter / 2
        volume = (4/3) * np.pi * radius**3
        mass = volume * density
        energy_joules = 0.5 * mass * velocity**2
        energy_megatons = energy_joules / (TNT_JOULE * 1000)
        return energy_joules, energy_megatons, mass
    
    @staticmethod
    def calculate_crater_size(energy_joules, target_type='sedimentary'):
        """Calculate crater diameter using scaling laws"""
        # Scaling constants based on target type
        scaling_constants = {
            'sedimentary': 1.3,
            'hard_rock': 1.0,
            'ice': 1.5
        }
        k = scaling_constants.get(target_type, 1.3)
        
        # Crater diameter in meters (simplified scaling law)
        crater_diameter = k * (energy_joules / 1e15)**(1/3.4)
        return crater_diameter
    
    @staticmethod
    def calculate_seismic_magnitude(energy_joules):
        """Calculate Richter scale magnitude"""
        # Gutenberg-Richter relation
        magnitude = (2/3) * np.log10(energy_joules) - 10.7
        return max(0, magnitude)
    
    @staticmethod
    def calculate_tsunami_height(energy_joules, distance_km, water_depth=4000):
        """Estimate tsunami wave height for ocean impacts"""
        if distance_km == 0:
            distance_km = 1
        
        # Simplified tsunami model
        energy_megatons = energy_joules / (TNT_JOULE * 1000)
        initial_height = 10 * (energy_megatons ** 0.5)
        
        # Wave height decreases with distance
        wave_height = initial_height * (100 / distance_km) ** 0.5
        return max(0.1, wave_height)
    
    @staticmethod
    def calculate_deflection(mass, velocity, delta_v, time_before_impact):
        """Calculate deflection distance from kinetic impactor"""
        # Momentum transfer
        momentum_change = mass * delta_v
        
        # Position change over time
        deflection_distance = delta_v * time_before_impact
        
        return deflection_distance

class NASADataFetcher:
    """Fetch real asteroid data from NASA APIs"""
    
    @staticmethod
    def get_neo_data(api_key='DEMO_KEY'):
        """Fetch Near-Earth Object data"""
        try:
            url = f"https://api.nasa.gov/neo/rest/v1/neo/browse?api_key={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    @staticmethod
    def get_sample_asteroids():
        """Return sample asteroid data for offline mode"""
        return [
            {
                'name': 'Impactor-2025',
                'diameter_m': 350,
                'velocity_km_s': 25,
                'a': 1.5,
                'e': 0.3,
                'i': 10,
                'omega': 45,
                'w': 120,
                'M': 30,
                'hazardous': True
            },
            {
                'name': 'Bennu',
                'diameter_m': 490,
                'velocity_km_s': 28,
                'a': 1.126,
                'e': 0.204,
                'i': 6.03,
                'omega': 2.06,
                'w': 66.22,
                'M': 101.7,
                'hazardous': True
            },
            {
                'name': 'Apophis',
                'diameter_m': 370,
                'velocity_km_s': 30.7,
                'a': 0.922,
                'e': 0.191,
                'i': 3.33,
                'omega': 204.4,
                'w': 126.4,
                'M': 246.5,
                'hazardous': True
            },
            {
                'name': 'Ryugu',
                'diameter_m': 900,
                'velocity_km_s': 20,
                'a': 1.19,
                'e': 0.19,
                'i': 5.88,
                'omega': 251.6,
                'w': 211.4,
                'M': 38.9,
                'hazardous': False
            }
        ]

def create_3d_trajectory_plot(asteroid_data, deflection_applied=False, deflection_params=None):
    """Create 3D visualization of asteroid trajectory"""
    physics = AsteroidPhysics()
    
    # Generate orbital path
    time_points = np.linspace(0, 365, 500)
    positions = []
    
    for t in time_points:
        x, y, z = physics.calculate_orbital_elements(
            asteroid_data['a'] * AU,
            asteroid_data['e'],
            asteroid_data['i'],
            asteroid_data['omega'],
            asteroid_data['w'],
            asteroid_data['M'] + t,
            t
        )
        positions.append([x/AU, y/AU, z/AU])
    
    positions = np.array(positions)
    
    # Create figure
    fig = go.Figure()
    
    # Add Sun
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=20, color='yellow', symbol='circle'),
        name='Sun',
        hovertext='Sun'
    ))
    
    # Add Earth orbit
    earth_orbit = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter3d(
        x=np.cos(earth_orbit),
        y=np.sin(earth_orbit),
        z=np.zeros(100),
        mode='lines',
        line=dict(color='cyan', width=2, dash='dash'),
        name='Earth Orbit'
    ))
    
    # Add Earth
    fig.add_trace(go.Scatter3d(
        x=[1], y=[0], z=[0],
        mode='markers',
        marker=dict(size=15, color='blue', symbol='circle'),
        name='Earth',
        hovertext='Earth'
    ))
    
    # Add asteroid trajectory
    color = 'red' if asteroid_data.get('hazardous', False) else 'green'
    if deflection_applied:
        color = 'orange'
    
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='lines+markers',
        line=dict(color=color, width=4),
        marker=dict(size=3),
        name=f"{asteroid_data['name']} Trajectory"
    ))
    
    # Add impact point or miss marker
    if not deflection_applied or (deflection_params and not deflection_params.get('success', True)):
        fig.add_trace(go.Scatter3d(
            x=[1], y=[0], z=[0],
            mode='markers',
            marker=dict(size=20, color='red', symbol='x'),
            name='Predicted Impact',
            hovertext='Collision Point'
        ))
    
    fig.update_layout(
        title=f"3D Orbital Trajectory: {asteroid_data['name']}",
        scene=dict(
            xaxis_title='X (AU)',
            yaxis_title='Y (AU)',
            zaxis_title='Z (AU)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='cube'
        ),
        height=600,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_impact_map(impact_location, crater_diameter, impact_type='land'):
    """Create 2D impact visualization map"""
    lat, lon = impact_location
    
    # Create circular impact zone
    angles = np.linspace(0, 2*np.pi, 100)
    
    # Convert crater diameter to degrees (rough approximation)
    radius_deg = (crater_diameter / 1000) / 111  # km to degrees
    
    impact_circle_lat = lat + radius_deg * np.cos(angles)
    impact_circle_lon = lon + radius_deg * np.sin(angles)
    
    # Create danger zones
    danger_zones = [
        {'radius': radius_deg * 5, 'color': 'rgba(255, 0, 0, 0.2)', 'name': 'Extreme Danger'},
        {'radius': radius_deg * 10, 'color': 'rgba(255, 100, 0, 0.15)', 'name': 'High Danger'},
        {'radius': radius_deg * 20, 'color': 'rgba(255, 200, 0, 0.1)', 'name': 'Moderate Danger'},
    ]
    
    fig = go.Figure()
    
    # Add danger zones
    for zone in danger_zones:
        zone_lat = lat + zone['radius'] * np.cos(angles)
        zone_lon = lon + zone['radius'] * np.sin(angles)
        
        fig.add_trace(go.Scattergeo(
            lat=zone_lat,
            lon=zone_lon,
            mode='lines',
            line=dict(width=0),
            fill='toself',
            fillcolor=zone['color'],
            name=zone['name'],
            hoverinfo='name'
        ))
    
    # Add crater
    fig.add_trace(go.Scattergeo(
        lat=impact_circle_lat,
        lon=impact_circle_lon,
        mode='lines',
        line=dict(width=3, color='red'),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.5)',
        name='Impact Crater',
        hoverinfo='name'
    ))
    
    # Add impact point
    fig.add_trace(go.Scattergeo(
        lat=[lat],
        lon=[lon],
        mode='markers',
        marker=dict(size=15, color='darkred', symbol='x'),
        name='Impact Center',
        hovertext=f'Impact Location: {lat:.2f}°, {lon:.2f}°'
    ))
    
    fig.update_geos(
        projection_type='natural earth',
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor='rgb(243, 243, 243)',
        coastlinecolor='rgb(204, 204, 204)',
        projection_scale=1.5 if impact_type == 'ocean' else 2,
        center=dict(lat=lat, lon=lon)
    )
    
    fig.update_layout(
        title='Predicted Impact Zone and Affected Areas',
        height=600,
        showlegend=True
    )
    
    return fig

def create_effects_dashboard(energy_megatons, crater_diameter, seismic_magnitude, 
                            tsunami_height=None, impact_type='land'):
    """Create comprehensive effects visualization"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Impact Energy",
            f"{energy_megatons:.2f} MT",
            help="Megatons of TNT equivalent"
        )
    
    with col2:
        st.metric(
            "Crater Diameter",
            f"{crater_diameter/1000:.2f} km",
            help="Expected crater size"
        )
    
    with col3:
        st.metric(
            "Seismic Magnitude",
            f"{seismic_magnitude:.1f}",
            help="Richter scale equivalent"
        )
    
    with col4:
        if tsunami_height and impact_type == 'ocean':
            st.metric(
                "Tsunami Height",
                f"{tsunami_height:.1f} m",
                help="Initial wave height"
            )
        else:
            st.metric(
                "Impact Type",
                impact_type.capitalize(),
                help="Location type"
            )
    
    # Comparison chart
    comparisons = {
        'Hiroshima (1945)': 0.015,
        'Tsar Bomba (1961)': 50,
        'Tunguska Event (1908)': 15,
        'Chicxulub (Dinosaurs)': 100_000_000,
        f'{st.session_state.current_asteroid}': energy_megatons
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(comparisons.keys()),
            y=list(comparisons.values()),
            marker_color=['gray', 'gray', 'orange', 'darkred', 'red'],
            text=[f"{v:.2e} MT" for v in comparisons.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Impact Energy Comparison',
        yaxis_type='log',
        yaxis_title='Energy (Megatons TNT)',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Effects timeline
    effects_data = {
        'Immediate (0-1 min)': [
            'Fireball visible from space',
            'Initial shockwave generation',
            'Crater formation begins'
        ],
        'Short-term (1 min - 1 hour)': [
            f'Seismic waves magnitude {seismic_magnitude:.1f}',
            'Ejecta dispersal begins',
            'Thermal radiation pulse'
        ],
        'Medium-term (1 hour - 1 day)': [
            'Regional devastation',
            'Fires from thermal radiation',
            'Tsunami propagation' if impact_type == 'ocean' else 'Dust cloud formation'
        ],
        'Long-term (1 day+)': [
            'Atmospheric effects',
            'Global temperature changes',
            'Ecological disruption'
        ]
    }
    
    st.subheader("Impact Effects Timeline")
    for period, effects in effects_data.items():
        with st.expander(period):
            for effect in effects:
                st.write(f"• {effect}")

def simulate_deflection_mission(asteroid_data, mission_params):
    """Simulate asteroid deflection mission"""
    physics = AsteroidPhysics()
    
    # Extract mission parameters
    mission_type = mission_params['type']
    launch_time = mission_params['launch_time']  # days before impact
    impactor_mass = mission_params.get('impactor_mass', 500)  # kg
    impactor_velocity = mission_params.get('impactor_velocity', 10000)  # m/s
    
    # Calculate asteroid properties
    energy_j, energy_mt, asteroid_mass = physics.calculate_impact_energy(
        asteroid_data['diameter_m'],
        asteroid_data['velocity_km_s'] * 1000
    )
    
    # Calculate deflection based on mission type
    if mission_type == 'kinetic_impactor':
        # Momentum transfer efficiency (beta factor, typically 1-4)
        beta = 1.9
        delta_v = (beta * impactor_mass * impactor_velocity) / asteroid_mass
        
        # Time available for deflection (convert days to seconds)
        time_available = launch_time * 24 * 3600
        
        deflection_distance = physics.calculate_deflection(
            asteroid_mass,
            asteroid_data['velocity_km_s'] * 1000,
            delta_v,
            time_available
        )
        
    elif mission_type == 'gravity_tractor':
        # Continuous low thrust over time
        spacecraft_mass = mission_params.get('spacecraft_mass', 1000)
        duration_days = launch_time * 0.8  # 80% of available time
        
        # Gravitational coupling
        distance = 100  # meters
        gravitational_force = G * spacecraft_mass * asteroid_mass / (distance ** 2)
        
        acceleration = gravitational_force / asteroid_mass
        time_seconds = duration_days * 24 * 3600
        delta_v = acceleration * time_seconds
        
        deflection_distance = 0.5 * acceleration * (time_seconds ** 2)
        
    elif mission_type == 'nuclear_device':
        # Standoff nuclear explosion
        explosive_yield = mission_params.get('yield_mt', 1) * TNT_JOULE * 1000
        standoff_distance = asteroid_data['diameter_m'] * 3
        
        # Energy transfer efficiency (simplified)
        efficiency = 0.2
        energy_absorbed = explosive_yield * efficiency
        
        delta_v = np.sqrt(2 * energy_absorbed / asteroid_mass)
        time_available = launch_time * 24 * 3600
        
        deflection_distance = delta_v * time_available
        
    else:  # laser_ablation
        laser_power = mission_params.get('laser_power_mw', 1)  # MW
        duration_days = launch_time * 0.5
        
        # Ablation thrust
        specific_impulse = 1000  # seconds
        mass_ablated = (laser_power * 1e6 * duration_days * 24 * 3600) / (specific_impulse * 9.81)
        
        delta_v = (mass_ablated * specific_impulse * 9.81) / asteroid_mass
        deflection_distance = delta_v * launch_time * 24 * 3600
    
    # Check if deflection is sufficient (need >6000 km deflection for miss)
    earth_radius_km = EARTH_RADIUS / 1000
    required_deflection = earth_radius_km * 1.5  # 1.5x Earth radius for safety
    
    success = (deflection_distance / 1000) > required_deflection
    miss_distance = deflection_distance / 1000 if success else -required_deflection
    
    return {
        'success': success,
        'deflection_distance_km': deflection_distance / 1000,
        'required_deflection_km': required_deflection,
        'delta_v': delta_v,
        'mission_type': mission_type,
        'launch_time_days': launch_time,
        'miss_distance_km': miss_distance
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">☄️ IMPACTOR-2025 DEFENSE SYSTEM ☄️</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #888; margin-bottom: 2rem;'>
        Advanced Asteroid Impact Simulation & Planetary Defense Platform
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.nasa.gov/wp-content/uploads/2023/03/nasa-logo-web-rgb.png?w=500", width=150)
        st.title("Mission Control")
        
        # Mode selection
        mode = st.radio(
            "Select Mode",
            ["🎯 Simulation Mode", "🛡️ Defense Mode", "📚 Educational Mode", "🌍 Real-Time Data"],
            help="Choose your mission type"
        )
        
        st.divider()
        
        # Asteroid selection
        asteroids = NASADataFetcher.get_sample_asteroids()
        asteroid_names = [a['name'] for a in asteroids]
        
        selected_asteroid_name = st.selectbox(
            "Select Asteroid",
            asteroid_names,
            help="Choose an asteroid to simulate"
        )
        
        selected_asteroid = next(a for a in asteroids if a['name'] == selected_asteroid_name)
        st.session_state.current_asteroid = selected_asteroid_name
        
        st.divider()
        
        # Display asteroid info
        st.subheader("Asteroid Properties")
        st.metric("Diameter", f"{selected_asteroid['diameter_m']} m")
        st.metric("Velocity", f"{selected_asteroid['velocity_km_s']} km/s")
        st.metric("Threat Level", "HIGH" if selected_asteroid['hazardous'] else "LOW")
        
        # Custom parameters
        with st.expander("⚙️ Custom Parameters"):
            custom_diameter = st.slider("Diameter (m)", 50, 2000, selected_asteroid['diameter_m'])
            custom_velocity = st.slider("Velocity (km/s)", 10, 50, selected_asteroid['velocity_km_s'])
            
            if st.button("Apply Custom"):
                selected_asteroid['diameter_m'] = custom_diameter
                selected_asteroid['velocity_km_s'] = custom_velocity
                st.success("Parameters updated!")
        
        st.divider()
        st.caption(f"Scenarios Completed: {st.session_state.scenarios_completed}")
        st.caption(f"Defense Score: {st.session_state.defense_score}")
    
    # Main content based on mode
    if mode == "🎯 Simulation Mode":
        simulation_mode(selected_asteroid)
    elif mode == "🛡️ Defense Mode":
        defense_mode(selected_asteroid)
    elif mode == "📚 Educational Mode":
        educational_mode(selected_asteroid)
    else:
        realtime_mode()

def simulation_mode(asteroid_data):
    """Interactive impact simulation"""
    st.header("Impact Simulation Laboratory")
    
    tabs = st.tabs(["🌍 Impact Scenario", "📊 Effects Analysis", "🗺️ Regional Impact"])
    
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Orbital Trajectory")
            trajectory_fig = create_3d_trajectory_plot(asteroid_data)
            st.plotly_chart(trajectory_fig, use_container_width=True)
        
        with col2:
            st.subheader("Impact Parameters")
            
            impact_lat = st.slider("Impact Latitude", -90.0, 90.0, 40.7, 0.1)
            impact_lon = st.slider("Impact Longitude", -180.0, 180.0, -74.0, 0.1)
            
            impact_type = st.selectbox(
                "Impact Location Type",
                ["land", "ocean", "coastal"],
                help="Affects secondary effects"
            )
            
            target_type = st.selectbox(
                "Surface Type",
                ["sedimentary", "hard_rock", "ice"],
                help="Affects crater formation"
            )
            
            impact_angle = st.slider("Impact Angle (degrees)", 15, 90, 45)
            
            if st.button("🔥 SIMULATE IMPACT", type="primary", use_container_width=True):
                st.session_state.simulation_run = True
                st.session_state.scenarios_completed += 1
    
    if st.session_state.simulation_run:
        st.divider()
        
        physics = AsteroidPhysics()
        
        # Calculate impact effects
        energy_j, energy_mt, mass = physics.calculate_impact_energy(
            asteroid_data['diameter_m'],
            asteroid_data['velocity_km_s'] * 1000
        )
        
        crater_diameter = physics.calculate_crater_size(energy_j, target_type)
        seismic_mag = physics.calculate_seismic_magnitude(energy_j)
        
        tsunami_height = None
        if impact_type == 'ocean':
            tsunami_height = physics.calculate_tsunami_height(energy_j, 100)
        
        with tabs[1]:
            st.subheader("📊 Impact Effects Analysis")
            create_effects_dashboard(
                energy_mt, crater_diameter, seismic_mag, tsunami_height, impact_type
            )
            
            # Additional analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Blast Radius")
                blast_radii = {
                    '100% Destruction': crater_diameter / 1000,
                    'Severe Damage': crater_diameter / 1000 * 2.5,
                    'Moderate Damage': crater_diameter / 1000 * 5,
                    'Light Damage': crater_diameter / 1000 * 10
                }
                
                for effect, radius in blast_radii.items():
                    st.write(f"**{effect}:** {radius:.1f} km")
            
            with col2:
                st.markdown("### Casualties Estimate")
                # Simplified casualty model
                if impact_type == 'ocean':
                    st.write("**Immediate:** Tsunami coastal areas")
                    st.write("**Short-term:** Coastal evacuations needed")
                    st.write("**Long-term:** Climate effects")
                else:
                    affected_area = np.pi * ((crater_diameter / 1000 * 10) ** 2)
                    avg_density = 50  # people per km²
                    casualties = int(affected_area * avg_density)
                    st.write(f"**Affected Area:** {affected_area:.0f} km²")
                    st.write(f"**Estimated Casualties:** {casualties:,}")
                    st.write(f"**Seismic Impact:** Magnitude {seismic_mag:.1f}")
        
        with tabs[2]:
            st.subheader("🗺️ Regional Impact Visualization")
            impact_map = create_impact_map(
                (impact_lat, impact_lon), 
                crater_diameter, 
                impact_type
            )
            st.plotly_chart(impact_map, use_container_width=True)
            
            # Affected cities
            st.markdown("### Major Cities in Affected Region")
            st.info("Cities within 1000 km of impact zone are at high risk from secondary effects.")
            
            # Environmental warnings
            if impact_type == 'ocean':
                st.warning("🌊 **TSUNAMI WARNING**: Coastal areas within 2000 km should evacuate immediately. Wave arrival time: 2-6 hours.")
            
            st.error("⚠️ **ATMOSPHERIC EFFECTS**: Dust and debris will affect global temperatures for 6-24 months.")

def defense_mode(asteroid_data):
    """Gamified planetary defense mode"""
    st.header("🛡️ Planetary Defense Command")
    
    st.markdown("""
    **Mission Brief**: An asteroid is on collision course with Earth. You must design and execute 
    a deflection mission to save humanity. Choose your strategy wisely - time and resources are limited!
    """)
    
    # Mission countdown
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Days Until Impact", "180", "-1 per day")
    with col2:
        st.metric("Mission Budget", "$2.5B", "±$500M")
    with col3:
        st.metric("Success Probability", "???", "Depends on strategy")
    
    st.divider()
    
    # Mission planning
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Mission Configuration")
        
        mission_type = st.selectbox(
            "Deflection Method",
            [
                "kinetic_impactor",
                "gravity_tractor", 
                "nuclear_device",
                "laser_ablation"
            ],
            format_func=lambda x: {
                "kinetic_impactor": "🚀 Kinetic Impactor (DART-style)",
                "gravity_tractor": "🛸 Gravity Tractor",
                "nuclear_device": "💣 Nuclear Standoff Burst",
                "laser_ablation": "⚡ Laser Ablation"
            }[x]
        )
        
        # Method-specific parameters
        st.markdown("### Mission Parameters")
        
        launch_time = st.slider(
            "Launch Time (days before impact)",
            30, 365, 180,
            help="Earlier launch = more deflection time"
        )
        
        mission_params = {
            'type': mission_type,
            'launch_time': launch_time
        }
        
        if mission_type == 'kinetic_impactor':
            st.markdown("**Kinetic Impactor Configuration**")
            impactor_mass = st.slider("Impactor Mass (kg)", 100, 1000, 500)
            impactor_velocity = st.slider("Impact Velocity (m/s)", 5000, 15000, 10000)
            mission_params['impactor_mass'] = impactor_mass
            mission_params['impactor_velocity'] = impactor_velocity
            
            st.info("💡 **Tip**: Higher mass and velocity = more momentum transfer. DART mission used 570 kg at ~6.6 km/s.")
            cost = 300 + (impactor_mass / 1000) * 200
            
        elif mission_type == 'gravity_tractor':
            st.markdown("**Gravity Tractor Configuration**")
            spacecraft_mass = st.slider("Spacecraft Mass (kg)", 500, 5000, 1000)
            mission_params['spacecraft_mass'] = spacecraft_mass
            
            st.info("💡 **Tip**: Requires long duration proximity operations. Gentle but slow deflection.")
            cost = 800 + (spacecraft_mass / 1000) * 100
            
        elif mission_type == 'nuclear_device':
            st.markdown("**Nuclear Device Configuration**")
            yield_mt = st.slider("Explosive Yield (Megatons)", 0.1, 5.0, 1.0)
            mission_params['yield_mt'] = yield_mt
            
            st.warning("⚠️ **Warning**: Requires international approval. Risk of fragmentation.")
            cost = 1500 + yield_mt * 200
            
        else:  # laser_ablation
            st.markdown("**Laser Ablation Configuration**")
            laser_power = st.slider("Laser Power (MW)", 0.1, 10.0, 1.0)
            mission_params['laser_power_mw'] = laser_power
            
            st.info("💡 **Tip**: Vaporizes surface material creating thrust. Requires precise targeting.")
            cost = 1000 + laser_power * 150
        
        st.metric("Mission Cost", f"${cost:.1f}M")
        
        execute_button = st.button("🚀 EXECUTE MISSION", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Asteroid Threat Assessment")
        
        physics = AsteroidPhysics()
        energy_j, energy_mt, mass = physics.calculate_impact_energy(
            asteroid_data['diameter_m'],
            asteroid_data['velocity_km_s'] * 1000
        )
        
        # Threat visualization
        threat_level = "EXTREME" if energy_mt > 1000 else "HIGH" if energy_mt > 100 else "MODERATE"
        threat_color = "red" if threat_level == "EXTREME" else "orange" if threat_level == "HIGH" else "yellow"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
            <h3>Threat Level: {threat_level}</h3>
            <p style='font-size: 1.2rem;'>Impact Energy: {energy_mt:.2f} Megatons</p>
            <p>Asteroid Mass: {mass/1e9:.2f} Million Tons</p>
            <p>Required Deflection: ~10,000 km</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display trajectory
        st.markdown("**Current Trajectory**")
        trajectory_fig = create_3d_trajectory_plot(asteroid_data)
        st.plotly_chart(trajectory_fig, use_container_width=True, key="defense_traj")
    
    # Execute mission simulation
    if execute_button:
        st.divider()
        st.subheader("🎯 Mission Execution Results")
        
        with st.spinner("Calculating mission trajectory and deflection..."):
            import time
            time.sleep(2)  # Dramatic pause
            
            result = simulate_deflection_mission(asteroid_data, mission_params)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Deflection Achieved",
                    f"{result['deflection_distance_km']:.1f} km",
                    f"{result['deflection_distance_km'] - result['required_deflection_km']:.1f} km"
                )
            
            with col2:
                st.metric(
                    "Delta-V Applied",
                    f"{result['delta_v']*1000:.2f} mm/s",
                    "Velocity change"
                )
            
            with col3:
                st.metric(
                    "Miss Distance",
                    f"{abs(result['miss_distance_km']):.1f} km",
                    "From Earth center"
                )
            
            st.divider()
            
            if result['success']:
                st.markdown("""
                <div class='defense-success'>
                    🎉 MISSION SUCCESS! 🎉<br>
                    Earth is safe! The asteroid will miss by a safe margin.
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
                st.session_state.defense_score += 100
                st.session_state.scenarios_completed += 1
                
                # Show deflected trajectory
                st.subheader("Updated Trajectory After Deflection")
                deflected_fig = create_3d_trajectory_plot(
                    asteroid_data, 
                    deflection_applied=True,
                    deflection_params=result
                )
                st.plotly_chart(deflected_fig, use_container_width=True)
                
                st.success(f"""
                **Mission Summary:**
                - Method: {result['mission_type'].replace('_', ' ').title()}
                - Launch: {result['launch_time_days']} days before impact
                - Deflection: {result['deflection_distance_km']:.1f} km
                - Final Status: **THREAT NEUTRALIZED**
                """)
                
            else:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); 
                padding: 20px; border-radius: 10px; color: white; text-align: center; 
                font-size: 1.5rem; font-weight: bold; margin: 20px 0;'>
                    ❌ MISSION FAILED ❌<br>
                    <span style='font-size: 1rem;'>Insufficient deflection - Impact imminent</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.error(f"""
                **Mission Analysis:**
                - Deflection Achieved: {result['deflection_distance_km']:.1f} km
                - Deflection Required: {result['required_deflection_km']:.1f} km
                - Shortfall: {result['required_deflection_km'] - result['deflection_distance_km']:.1f} km
                
                **Recommendations:**
                - Launch earlier for more deflection time
                - Increase impactor mass/velocity
                - Consider alternative deflection method
                - Launch multiple missions
                """)
                
                st.info("💡 Try adjusting your mission parameters and try again!")

def educational_mode(asteroid_data):
    """Educational content about asteroids and planetary defense"""
    st.header("📚 Asteroid Defense Academy")
    
    tabs = st.tabs([
        "🌌 Asteroid Basics",
        "🔬 Impact Physics", 
        "🛡️ Defense Methods",
        "📜 Historical Impacts",
        "🎓 Quiz Mode"
    ])
    
    with tabs[0]:
        st.subheader("Understanding Near-Earth Asteroids")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### What are Near-Earth Asteroids?
            
            Near-Earth Asteroids (NEAs) are rocky bodies that orbit the Sun and come within 
            1.3 AU of Earth's orbit. They are remnants from the formation of our solar system 
            4.6 billion years ago.
            
            **Classification by Orbit:**
            - **Atens**: Orbit mostly inside Earth's orbit (a < 1.0 AU)
            - **Apollos**: Cross Earth's orbit from outside (a > 1.0 AU)
            - **Amors**: Approach but don't cross Earth's orbit
            
            **Size Categories:**
            - Small: < 25 m (frequent, mostly burn up)
            - Medium: 25-140 m (Tunguska-scale)
            - Large: 140 m - 1 km (regional devastation)
            - Catastrophic: > 1 km (global effects)
            """)
            
            # Interactive size comparison
            sizes = {
                'Impactor-2025': asteroid_data['diameter_m'],
                'Statue of Liberty': 93,
                'Eiffel Tower': 330,
                'Empire State Building': 443,
                'Football Field': 110,
                'Bus': 12
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    y=list(sizes.keys()),
                    x=list(sizes.values()),
                    orientation='h',
                    marker_color='teal',
                    text=[f"{v} m" for v in sizes.values()],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='Asteroid Size Comparison',
                xaxis_title='Size (meters)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            ### Orbital Mechanics
            
            Asteroids follow elliptical orbits around the Sun, described by **Keplerian elements**:
            
            1. **Semi-major axis (a)**: Average distance from Sun
            2. **Eccentricity (e)**: How elliptical the orbit is (0 = circle, 0.9 = very elongated)
            3. **Inclination (i)**: Tilt relative to Earth's orbital plane
            4. **Longitude of ascending node (Ω)**: Where orbit crosses Earth's plane
            5. **Argument of perihelion (ω)**: Orientation of orbit's closest point
            6. **Mean anomaly (M)**: Position along orbit at a given time
            """)
            
            # Show current asteroid's orbital elements
            st.markdown(f"""
            ### {asteroid_data['name']} Orbital Elements
            
            - **Semi-major axis**: {asteroid_data['a']} AU
            - **Eccentricity**: {asteroid_data['e']}
            - **Inclination**: {asteroid_data['i']}°
            - **Long. of Asc. Node**: {asteroid_data['omega']}°
            - **Arg. of Perihelion**: {asteroid_data['w']}°
            - **Mean Anomaly**: {asteroid_data['M']}°
            """)
            
            # Visualization of orbital elements
            st.info("""
            💡 **Did you know?** 
            - There are over 30,000 known Near-Earth Asteroids
            - NASA tracks all NEAs larger than 140 meters
            - About 50 new NEAs are discovered each week
            - The chance of a major impact in our lifetime is very low (~0.01%)
            """)
    
    with tabs[1]:
        st.subheader("The Physics of Asteroid Impacts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Impact Energy Calculation
            
            The kinetic energy of an asteroid impact is calculated using:
            
            **E = ½ × m × v²**
            
            Where:
            - **E** = Kinetic energy (Joules)
            - **m** = Asteroid mass (kg)
            - **v** = Impact velocity (m/s)
            
            **Mass Calculation:**
            1. Volume = (4/3) × π × r³
            2. Mass = Volume × Density
            3. Typical asteroid density: 3000 kg/m³
            
            **Energy Conversion:**
            - 1 Megaton TNT = 4.184 × 10¹⁵ Joules
            """)
            
            # Interactive calculator
            st.markdown("### 🧮 Impact Energy Calculator")
            calc_diameter = st.number_input("Asteroid Diameter (m)", 10, 1000, 100)
            calc_velocity = st.number_input("Impact Velocity (km/s)", 10, 70, 20)
            calc_density = st.number_input("Density (kg/m³)", 2000, 5000, 3000)
            
            if st.button("Calculate Impact Energy"):
                physics = AsteroidPhysics()
                energy_j, energy_mt, mass = physics.calculate_impact_energy(
                    calc_diameter,
                    calc_velocity * 1000,
                    calc_density
                )
                
                st.success(f"""
                **Results:**
                - Asteroid Mass: {mass/1e9:.2f} billion kg
                - Impact Energy: {energy_mt:.2f} Megatons TNT
                - Equivalent to {energy_mt/0.015:.0f} Hiroshima bombs
                """)
        
        with col2:
            st.markdown("""
            ### Crater Formation
            
            Impact craters form through a complex process:
            
            **Stages:**
            1. **Contact/Compression** (microseconds): Shockwave propagates
            2. **Excavation** (seconds): Material ejected, crater opens
            3. **Modification** (minutes): Crater walls collapse, central peak forms
            
            **Crater Scaling Laws:**
            
            Crater diameter ∝ Energy^(1/3.4)
            
            **Factors Affecting Crater Size:**
            - Impact energy (most important)
            - Impact angle (optimal: 45°)
            - Target material (soft rock > hard rock)
            - Gravity (lower gravity = bigger craters)
            """)
            
            # Crater size visualization
            energies = np.logspace(12, 24, 50)
            crater_sizes = [AsteroidPhysics.calculate_crater_size(e) for e in energies]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=energies / 1e15,
                y=np.array(crater_sizes) / 1000,
                mode='lines',
                line=dict(color='orange', width=3),
                fill='tozeroy'
            ))
            
            fig.update_layout(
                title='Crater Size vs Impact Energy',
                xaxis_title='Energy (Petajoules)',
                yaxis_title='Crater Diameter (km)',
                xaxis_type='log',
                yaxis_type='log',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            📊 **Crater Examples:**
            - Meteor Crater, Arizona: 1.2 km (50,000 years old)
            - Chicxulub, Mexico: 150 km (dinosaur extinction)
            - Vredefort, South Africa: 300 km (largest confirmed, 2 billion years old)
            """)
    
    with tabs[2]:
        st.subheader("Planetary Defense Strategies")
        
        defense_methods = {
            "🚀 Kinetic Impactor": {
                "description": "Smash a spacecraft into the asteroid to change its velocity",
                "pros": [
                    "Proven technology (DART mission 2022)",
                    "Relatively simple and reliable",
                    "Can be launched quickly",
                    "Good for medium-sized asteroids"
                ],
                "cons": [
                    "Requires years of warning time",
                    "Risk of fragmentation",
                    "Limited deflection per mission",
                    "Requires accurate targeting"
                ],
                "status": "✅ OPERATIONAL",
                "example": "NASA's DART mission successfully changed Dimorphos' orbit by 33 minutes",
                "timeline": "10-20 years warning needed"
            },
            "🛸 Gravity Tractor": {
                "description": "Park a spacecraft near the asteroid and use gravitational attraction to slowly pull it off course",
                "pros": [
                    "Very gentle - no fragmentation risk",
                    "Precise control of deflection",
                    "Can handle irregular asteroids",
                    "Predictable results"
                ],
                "cons": [
                    "Extremely slow - decades needed",
                    "Requires precise station-keeping",
                    "High fuel requirements",
                    "Expensive long-duration mission"
                ],
                "status": "🔬 THEORETICAL",
                "example": "Proposed for 99942 Apophis if needed",
                "timeline": "20-50 years warning needed"
            },
            "💣 Nuclear Deflection": {
                "description": "Detonate a nuclear device near (not on) the asteroid to vaporize surface material and create thrust",
                "pros": [
                    "Most powerful option available",
                    "Can handle large asteroids",
                    "Works with short warning times",
                    "Effective for difficult cases"
                ],
                "cons": [
                    "Risk of fragmenting asteroid",
                    "Requires international cooperation",
                    "Political complications",
                    "Unpredictable secondary effects"
                ],
                "status": "⚠️ LAST RESORT",
                "example": "Proposed for asteroids >1 km with <10 years warning",
                "timeline": "5-15 years warning needed"
            },
            "⚡ Laser Ablation": {
                "description": "Use high-power lasers to vaporize surface material, creating a rocket-like thrust",
                "pros": [
                    "Very precise control",
                    "Can work continuously",
                    "No consumables needed (solar powered)",
                    "Minimal fragmentation risk"
                ],
                "cons": [
                    "Technology not yet mature",
                    "Requires very long duration",
                    "Power requirements are huge",
                    "Only for small asteroids"
                ],
                "status": "🔬 EXPERIMENTAL",
                "example": "DE-STAR concept proposes orbital laser array",
                "timeline": "30-50 years warning needed"
            },
            "🎯 Ion Beam Shepherd": {
                "description": "Use ion beam from a spacecraft to push the asteroid gradually",
                "pros": [
                    "Gentle and controlled",
                    "Can fine-tune deflection",
                    "No physical contact needed",
                    "Scalable with multiple craft"
                ],
                "cons": [
                    "Very slow process",
                    "Complex spacecraft required",
                    "Limited by power available",
                    "Decades of operation needed"
                ],
                "status": "🔬 CONCEPT",
                "example": "Proposed as Apophis backup option",
                "timeline": "25-40 years warning needed"
            },
            "⚓ Mass Driver": {
                "description": "Land on asteroid, mine material, and launch it to create reactive thrust",
                "pros": [
                    "Uses asteroid's own mass",
                    "Very efficient long-term",
                    "Can significantly change orbit",
                    "Good for resource-rich asteroids"
                ],
                "cons": [
                    "Extremely complex mission",
                    "Requires landing and mining",
                    "Very long timeline needed",
                    "Untested technology"
                ],
                "status": "🔬 THEORETICAL",
                "example": "Proposed for asteroid redirect missions",
                "timeline": "40+ years warning needed"
            }
        }
        
        for method, details in defense_methods.items():
            with st.expander(f"{method} - {details['status']}"):
                st.markdown(f"### {details['description']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ✅ Advantages")
                    for pro in details['pros']:
                        st.write(f"- {pro}")
                
                with col2:
                    st.markdown("#### ❌ Disadvantages")
                    for con in details['cons']:
                        st.write(f"- {con}")
                
                st.info(f"**Example:** {details['example']}")
                st.warning(f"**Warning Time Required:** {details['timeline']}")
        
        st.divider()
        
        st.markdown("""
        ### 🌍 International Cooperation
        
        Planetary defense requires global coordination:
        
        - **NASA Planetary Defense Coordination Office (PDCO)**: US lead agency
        - **ESA Space Situational Awareness**: European monitoring
        - **IAWN (International Asteroid Warning Network)**: Global early warning
        - **SMPAG (Space Mission Planning Advisory Group)**: Mission coordination
        - **UN Committee on Peaceful Uses of Outer Space**: Policy framework
        """)
    
    with tabs[3]:
        st.subheader("📜 Historical Impact Events")
        
        historical_impacts = [
            {
                "name": "Chicxulub Impact",
                "date": "66 million years ago",
                "location": "Yucatan Peninsula, Mexico",
                "size": "10-15 km diameter",
                "energy": "100,000,000 MT",
                "effects": "Caused mass extinction including dinosaurs. Global firestorms, tsunamis, and impact winter lasting years.",
                "crater": "180 km diameter crater"
            },
            {
                "name": "Tunguska Event",
                "date": "June 30, 1908",
                "location": "Siberia, Russia",
                "size": "50-60 m diameter",
                "energy": "10-15 MT",
                "effects": "Flattened 2,000 km² of forest. Airburst at 5-10 km altitude. No crater formed. Heard 1,000 km away.",
                "crater": "None (airburst)"
            },
            {
                "name": "Barringer (Meteor) Crater",
                "date": "50,000 years ago",
                "location": "Arizona, USA",
                "size": "50 m diameter",
                "energy": "10 MT",
                "effects": "Created famous tourist crater. Impact equivalent to large nuclear weapon.",
                "crater": "1.2 km diameter, 170 m deep"
            },
            {
                "name": "Chelyabinsk Meteor",
                "date": "February 15, 2013",
                "location": "Chelyabinsk, Russia",
                "size": "20 m diameter",
                "energy": "0.5 MT",
                "effects": "Airburst at 30 km altitude. Shockwave damaged 7,200 buildings. 1,500 people injured by broken glass.",
                "crater": "6 m hole in frozen lake"
            },
            {
                "name": "Vredefort Impact",
                "date": "2 billion years ago",
                "location": "South Africa",
                "size": "10-15 km diameter",
                "energy": "100,000,000+ MT",
                "effects": "Largest verified impact structure on Earth. Original crater eroded over billions of years.",
                "crater": "300 km original diameter"
            },
            {
                "name": "Sudbury Basin",
                "date": "1.8 billion years ago",
                "location": "Ontario, Canada",
                "size": "10-15 km diameter",
                "energy": "~100,000,000 MT",
                "effects": "Second-largest impact structure. Rich mineral deposits formed. Major nickel mining region.",
                "crater": "250 km diameter"
            }
        ]
        
        # Timeline visualization
        dates = [-66000000, -2000000000, -1800000000, -50000, -115, -0]
        names = ["Chicxulub", "Vredefort", "Sudbury", "Barringer", "Tunguska", "Chelyabinsk"]
        sizes = [15000, 15000, 15000, 50, 60, 20]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=sizes,
            mode='markers+text',
            marker=dict(
                size=[30, 30, 30, 20, 20, 15],
                color=['darkred', 'purple', 'purple', 'orange', 'orange', 'yellow'],
                line=dict(width=2, color='white')
            ),
            text=names,
            textposition="top center",
            hovertext=[f"{name}<br>Size: {size}m" for name, size in zip(names, sizes)],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='Major Impact Events Timeline',
            xaxis_title='Years Ago (log scale)',
            yaxis_title='Impactor Size (meters, log scale)',
            xaxis_type='log',
            yaxis_type='log',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed cards for each event
        for impact in historical_impacts:
            with st.expander(f"💥 {impact['name']} - {impact['date']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **Location:** {impact['location']}
                    
                    **Impactor Size:** {impact['size']}
                    
                    **Energy Released:** {impact['energy']}
                    
                    **Effects:** {impact['effects']}
                    """)
                
                with col2:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 20px; border-radius: 10px; color: white;'>
                        <h4>Crater Info</h4>
                        <p>{impact['crater']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.info("""
        📊 **Impact Frequency:**
        - 10 m objects: Every 10 years (mostly burn up)
        - 50 m objects: Every 1,000 years (Tunguska-scale)
        - 1 km objects: Every 500,000 years (mass extinction potential)
        - 10 km objects: Every 100 million years (global catastrophe)
        """)
    
    with tabs[4]:
        st.subheader("🎓 Test Your Knowledge")
        
        st.markdown("### Asteroid Defense Quiz")
        
        questions = [
            {
                "question": "What does 'NEO' stand for?",
                "options": ["Near-Earth Object", "New Earth Orbit", "Nuclear Explosion Outcome", "Next Event Occurrence"],
                "correct": 0,
                "explanation": "NEO stands for Near-Earth Object, referring to asteroids and comets that come within 1.3 AU of Earth."
            },
            {
                "question": "Which NASA mission successfully demonstrated asteroid deflection in 2022?",
                "options": ["OSIRIS-REx", "DART", "Hayabusa2", "Dawn"],
                "correct": 1,
                "explanation": "DART (Double Asteroid Redirection Test) successfully impacted Dimorphos in September 2022, changing its orbit."
            },
            {
                "question": "What caused the extinction of the dinosaurs?",
                "options": ["Volcanic eruption", "Ice age", "Asteroid impact", "Disease"],
                "correct": 2,
                "explanation": "The Chicxulub asteroid impact 66 million years ago caused the mass extinction that killed the dinosaurs."
            },
            {
                "question": "How much warning time is typically needed for a kinetic impactor mission?",
                "options": ["1-2 years", "5-10 years", "10-20 years", "50+ years"],
                "correct": 2,
                "explanation": "Kinetic impactor missions need 10-20 years warning to be effective, allowing time for planning, launch, and deflection."
            },
            {
                "question": "What was the Tunguska event?",
                "options": ["Nuclear test", "Volcano", "Asteroid airburst", "Earthquake"],
                "correct": 2,
                "explanation": "The 1908 Tunguska event was an asteroid airburst that flattened 2,000 km² of Siberian forest."
            },
            {
                "question": "Which deflection method uses gravitational attraction?",
                "options": ["Kinetic impactor", "Gravity tractor", "Nuclear device", "Laser ablation"],
                "correct": 1,
                "explanation": "A gravity tractor uses a spacecraft's gravitational pull to slowly tug an asteroid off course."
            },
            {
                "question": "What is the typical density of an asteroid?",
                "options": ["1,000 kg/m³", "3,000 kg/m³", "8,000 kg/m³", "20,000 kg/m³"],
                "correct": 1,
                "explanation": "Most asteroids have a density around 3,000 kg/m³, similar to rocky materials on Earth."
            },
            {
                "question": "How often do Tunguska-scale impacts occur on Earth?",
                "options": ["Every 10 years", "Every 100 years", "Every 1,000 years", "Every 10,000 years"],
                "correct": 2,
                "explanation": "Events like Tunguska (50m objects) occur roughly every 1,000 years on average."
            }
        ]
        
        if 'quiz_score' not in st.session_state:
            st.session_state.quiz_score = 0
            st.session_state.quiz_answers = {}
        
        for i, q in enumerate(questions):
            st.markdown(f"**Question {i+1}: {q['question']}**")
            
            answer = st.radio(
                f"Select your answer for Q{i+1}:",
                q['options'],
                key=f"quiz_{i}",
                label_visibility="collapsed"
            )
            
            if st.button(f"Submit Answer {i+1}", key=f"submit_{i}"):
                selected_index = q['options'].index(answer)
                
                if selected_index == q['correct']:
                    st.success(f"✅ Correct! {q['explanation']}")
                    if i not in st.session_state.quiz_answers:
                        st.session_state.quiz_score += 1
                        st.session_state.quiz_answers[i] = True
                else:
                    st.error(f"❌ Incorrect. The correct answer is: {q['options'][q['correct']]}. {q['explanation']}")
                    st.session_state.quiz_answers[i] = False
            
            st.divider()
        
        if len(st.session_state.quiz_answers) == len(questions):
            score_percent = (st.session_state.quiz_score / len(questions)) * 100
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px; border-radius: 15px; color: white; text-align: center; margin: 20px 0;'>
                <h2>Final Score: {st.session_state.quiz_score}/{len(questions)}</h2>
                <h3>{score_percent:.0f}%</h3>
                <p style='font-size: 1.2rem;'>
                    {
                        "🏆 Planetary Defense Expert!" if score_percent >= 90 else
                        "🌟 Asteroid Hunter!" if score_percent >= 70 else
                        "📚 Keep Learning!" if score_percent >= 50 else
                        "🚀 Study More!"
                    }
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Reset Quiz"):
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = {}
                st.rerun()

def realtime_mode():
    """Fetch and display real-time asteroid data"""
    st.header("🌍 Real-Time Near-Earth Object Data")
    
    st.info("🔴 **LIVE DATA**: Connecting to NASA's Near-Earth Object API...")
    
    # API key input
    api_key = st.text_input(
        "NASA API Key (or use DEMO_KEY for limited access)",
        value="DEMO_KEY",
        type="password",
        help="Get your free API key at https://api.nasa.gov"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        This mode fetches real-time data about Near-Earth Objects from NASA's API.
        Data includes asteroid size, velocity, close approach dates, and hazard assessment.
        """)
    
    with col2:
        fetch_button = st.button("🔄 Fetch Live Data", type="primary", use_container_width=True)
    
    if fetch_button:
        with st.spinner("Fetching data from NASA..."):
            neo_data = NASADataFetcher.get_neo_data(api_key)
            
            if neo_data and 'near_earth_objects' in neo_data:
                st.success("✅ Successfully connected to NASA API!")
                
                # Parse NEO data
                neos = neo_data['near_earth_objects']
                
                # Create DataFrame
                neo_list = []
                for neo in neos:
                    neo_list.append({
                        'Name': neo.get('name', 'Unknown'),
                        'Diameter (m)': neo.get('estimated_diameter', {}).get('meters', {}).get('estimated_diameter_max', 0),
                        'Hazardous': 'Yes' if neo.get('is_potentially_hazardous_asteroid', False) else 'No',
                        'Close Approach': neo.get('close_approach_data', [{}])[0].get('close_approach_date', 'N/A') if neo.get('close_approach_data') else 'N/A',
                        'Velocity (km/s)': float(neo.get('close_approach_data', [{}])[0].get('relative_velocity', {}).get('kilometers_per_second', 0)) if neo.get('close_approach_data') else 0,
                        'Miss Distance (km)': float(neo.get('close_approach_data', [{}])[0].get('miss_distance', {}).get('kilometers', 0)) if neo.get('close_approach_data') else 0
                    })
                
                df = pd.DataFrame(neo_list)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total NEOs Tracked", len(df))
                
                with col2:
                    hazardous_count = len(df[df['Hazardous'] == 'Yes'])
                    st.metric("Potentially Hazardous", hazardous_count, 
                             delta=f"{(hazardous_count/len(df)*100):.1f}%")
                
                with col3:
                    avg_size = df['Diameter (m)'].mean()
                    st.metric("Average Diameter", f"{avg_size:.1f} m")
                
                with col4:
                    max_size = df['Diameter (m)'].max()
                    st.metric("Largest Asteroid", f"{max_size:.1f} m")
                
                st.divider()
                
                # Visualization tabs
                tab1, tab2, tab3 = st.tabs(["📊 Data Table", "📈 Analytics", "🎯 Threat Assessment"])
                
                with tab1:
                    st.subheader("Near-Earth Objects Database")
                    
                    # Filters
                    col1, col2 = st.columns(2)
                    with col1:
                        show_hazardous_only = st.checkbox("Show only potentially hazardous asteroids")
                    with col2:
                        min_size = st.slider("Minimum diameter (m)", 0, int(df['Diameter (m)'].max()), 0)
                    
                    filtered_df = df.copy()
                    if show_hazardous_only:
                        filtered_df = filtered_df[filtered_df['Hazardous'] == 'Yes']
                    filtered_df = filtered_df[filtered_df['Diameter (m)'] >= min_size]
                    
                    st.dataframe(
                        filtered_df.style.apply(
                            lambda x: ['background-color: #ffcccc' if v == 'Yes' else '' for v in x], 
                            subset=['Hazardous']
                        ),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download button
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Data as CSV",
                        csv,
                        "neo_data.csv",
                        "text/csv",
                        key='download-csv'
                    )
                
                with tab2:
                    st.subheader("NEO Analytics Dashboard")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Size distribution
                        fig = px.histogram(
                            df,
                            x='Diameter (m)',
                            nbins=30,
                            title='Asteroid Size Distribution',
                            color='Hazardous',
                            color_discrete_map={'Yes': 'red', 'No': 'green'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Velocity distribution
                        fig = px.box(
                            df,
                            y='Velocity (km/s)',
                            x='Hazardous',
                            title='Velocity Distribution by Threat Level',
                            color='Hazardous',
                            color_discrete_map={'Yes': 'red', 'No': 'green'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Size vs Velocity scatter
                    fig = px.scatter(
                        df,
                        x='Diameter (m)',
                        y='Velocity (km/s)',
                        size='Miss Distance (km)',
                        color='Hazardous',
                        hover_data=['Name'],
                        title='Asteroid Characteristics: Size vs Velocity',
                        color_discrete_map={'Yes': 'red', 'No': 'green'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    st.subheader("🎯 Threat Assessment Matrix")
                    
                    # Calculate threat scores
                    df['Threat Score'] = (
                        (df['Diameter (m)'] / df['Diameter (m)'].max()) * 0.4 +
                        (df['Velocity (km/s)'] / df['Velocity (km/s)'].max()) * 0.3 +
                        (1 - df['Miss Distance (km)'] / df['Miss Distance (km)'].max()) * 0.3
                    ) * 100
                    
                    df['Threat Level'] = pd.cut(
                        df['Threat Score'],
                        bins=[0, 25, 50, 75, 100],
                        labels=['Low', 'Moderate', 'High', 'Extreme']
                    )
                    
                    # Top threats
                    st.markdown("### 🚨 Top 10 Highest Threat Objects")
                    
                    top_threats = df.nlargest(10, 'Threat Score')[
                        ['Name', 'Diameter (m)', 'Velocity (km/s)', 'Miss Distance (km)', 'Threat Score', 'Threat Level']
                    ]
                    
                    st.dataframe(
                        top_threats.style.background_gradient(subset=['Threat Score'], cmap='Reds'),
                        use_container_width=True
                    )
                    
                    # Threat level distribution
                    threat_counts = df['Threat Level'].value_counts()
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=threat_counts.index,
                            values=threat_counts.values,
                            marker=dict(colors=['green', 'yellow', 'orange', 'red']),
                            hole=0.4
                        )
                    ])
                    
                    fig.update_layout(
                        title='Threat Level Distribution',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Close approach timeline
                    df_with_dates = df[df['Close Approach'] != 'N/A'].copy()
                    if not df_with_dates.empty:
                        df_with_dates['Close Approach Date'] = pd.to_datetime(df_with_dates['Close Approach'])
                        df_with_dates = df_with_dates.sort_values('Close Approach Date')
                        
                        fig = go.Figure()
                        
                        for idx, row in df_with_dates.iterrows():
                            color = 'red' if row['Hazardous'] == 'Yes' else 'green'
                            fig.add_trace(go.Scatter(
                                x=[row['Close Approach Date']],
                                y=[row['Miss Distance (km)']],
                                mode='markers',
                                marker=dict(
                                    size=row['Diameter (m)']/10,
                                    color=color,
                                    line=dict(width=2, color='white')
                                ),
                                name=row['Name'],
                                hovertext=f"{row['Name']}<br>Size: {row['Diameter (m)']:.0f}m<br>Date: {row['Close Approach']}"
                            ))
                        
                        fig.update_layout(
                            title='Close Approach Timeline',
                            xaxis_title='Date',
                            yaxis_title='Miss Distance (km)',
                            yaxis_type='log',
                            height=500,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("❌ Failed to fetch data from NASA API. Using sample data instead.")
                st.info("💡 This might be due to API rate limits with DEMO_KEY. Get your free API key at https://api.nasa.gov")
                
                # Show sample data
                sample_asteroids = NASADataFetcher.get_sample_asteroids()
                df_sample = pd.DataFrame(sample_asteroids)
                st.dataframe(df_sample, use_container_width=True)
    
    # Additional resources
    st.divider()
    st.subheader("📚 Additional Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **NASA Resources:**
        - [NASA CNEOS](https://cneos.jpl.nasa.gov/)
        - [NEO Close Approaches](https://cneos.jpl.nasa.gov/ca/)
        - [Sentry Impact Risk](https://cneos.jpl.nasa.gov/sentry/)
        """)
    
    with col2:
        st.markdown("""
        **Live Tracking:**
        - [Eyes on Asteroids](https://eyes.nasa.gov/apps/asteroids/)
        - [Minor Planet Center](https://minorplanetcenter.net/)
        - [IAU Minor Planet Center](https://www.minorplanetcenter.net/iau/lists/Closest.html)
        """)
    
    with col3:
        st.markdown("""
        **Defense Programs:**
        - [Planetary Defense](https://www.nasa.gov/planetarydefense/)
        - [DART Mission](https://www.nasa.gov/dart)
        - [NEO Surveyor](https://www.jpl.nasa.gov/missions/neo-surveyor)
        """)

# Run the main application
if __name__ == "__main__":
    main()

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Impactor-2025 Defense System</strong></p>
        <p>Powered by NASA NEO API | USGS Data | Advanced Orbital Mechanics</p>
        <p>🌍 Protecting Earth, One Simulation at a Time 🛡️</p>
        <p style='font-size: 0.9rem; margin-top: 10px;'>
            Data Sources: NASA CNEOS, JPL Small-Body Database, USGS National Map
        </p>
        <p style='font-size: 0.8rem; margin-top: 5px;'>
            Educational Tool | Not for Actual Defense Planning
        </p>
    </div>
    """, unsafe_allow_html=True)
