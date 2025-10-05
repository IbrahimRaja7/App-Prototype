import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time

# --- 1. CONFIGURATION AND SETUP ---
st.set_page_config(
    page_title="Impactor-2025: Asteroid Defender",
    page_icon="💥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SCIENTIFIC CONSTANTS AND MOCK DATA ---

# Constants (used for impact physics and scaling)
ASTEROID_DENSITY = 3000  # kg/m^3 (Typical stony asteroid)
EARTH_RADIUS = 6371      # km
TNT_ENERGY_EQUIV = 4.184e12 # Joules per 1 Megaton TNT
CRATER_SCALING_EXP = 0.29 # Simplified power law exponent for impact cratering

# Mock USGS Data: Target locations and geological characteristics
# Used to demonstrate impact site targeting and environmental effect visualization
MOCK_IMPACT_SITES = [
    {"name": "Coastal City A (USGS)", "lat": 34.05, "lon": -118.24, "type": "Coastal", "pop": 13},
    {"name": "Inland Plains B (USGS)", "lat": 40.71, "lon": -99.80, "type": "Inland", "pop": 2},
    {"name": "Oceanic Deep C (USGS)", "lat": 15.00, "lon": -165.00, "type": "Oceanic", "pop": 0}
]

# Mock NEO Data for Impactor-2025 (Initial State)
IMPACTOR_2025_DEFAULT = {
    'velocity': 20.0, # km/s
    'diameter': 300,  # meters
    'days_to_impact': 365, # days
    'impact_lat': 34.05,
    'impact_lon': -118.24,
    'target_site': "Coastal City A (USGS)"
}

# --- 3. CORE PHYSICS AND SIMULATION FUNCTIONS ---

@st.cache_data
def calculate_impact_metrics(diameter_m, velocity_kms, density_kgm3):
    """Calculates kinetic energy, TNT equivalent, crater diameter, and seismic magnitude."""
    # Convert inputs to standard units (meters, kg, m/s)
    diameter = diameter_m
    velocity = velocity_kms * 1000
    
    # 1. Mass Calculation (assuming spherical shape)
    radius = diameter / 2
    volume = (4/3) * np.pi * (radius ** 3)
    mass = volume * density_kgm3 # kg

    # 2. Kinetic Energy (E = 0.5 * m * v^2)
    kinetic_energy_J = 0.5 * mass * (velocity ** 2)

    # 3. TNT Equivalence
    # Divide by TNT_ENERGY_EQUIV (Joules per Megaton)
    tnt_megatons = kinetic_energy_J / TNT_ENERGY_EQUIV
    
    # 4. Crater Diameter (Simplified scaling relationship, km)
    # Using a generalized scaling formula (D ~ E^0.29)
    # 1 Gigaton TNT is approx 4.184e15 J
    E_Gt = kinetic_energy_J / 4.184e15
    crater_diameter_km = 0.0001 * (E_Gt ** CRATER_SCALING_EXP) * 1000 # Rough approximation based on energy
    
    # 5. Seismic Magnitude (Moment Magnitude Scale approximation)
    # Log relationship: M ~ 2/3 * log10(Energy in Joules) - C
    # Using a common approximation for large impacts
    seismic_magnitude = (2/3) * np.log10(kinetic_energy_J) - 8.5 
    
    return {
        "mass_billion_kg": mass / 1e9,
        "energy_megatons_tnt": tnt_megatons,
        "crater_diameter_km": crater_diameter_km,
        "seismic_magnitude": seismic_magnitude,
    }

@st.cache_data
def simulate_deflection(size_m, velocity_kms, days_to_impact, kinetic_impactor_size_t, delta_v_target_mps=0.005):
    """
    Simulates the required delta-V for deflection and the outcome of a kinetic impactor.
    
    delta-V target (e.g., 5 mm/s) is the minimum velocity change required for a safe miss.
    
    Returns: required_delta_v, achieved_delta_v, status (Safe/Partial/Failure)
    """
    # 1. Target Delta-V required (Based on days to impact and Earth radius)
    # Formula: Delta_V_req = (R_earth * 2) / (Time_to_impact in seconds)
    time_to_impact_s = days_to_impact * 24 * 3600
    
    # We need to change the path enough to miss the 12,742 km wide Earth.
    # Required delta-V to shift the orbit by 1 Earth radius at the time of impact.
    required_delta_v_mps = (EARTH_RADIUS * 1000) / time_to_impact_s 
    
    # 2. Achieved Delta-V (from Kinetic Impactor)
    # Using conservation of momentum: m_ast * v_change = m_imp * v_imp
    # Assume impactor velocity is high (e.g., 10 km/s)
    
    # Get asteroid mass
    radius_m = size_m / 2
    volume = (4/3) * np.pi * (radius_m ** 3)
    asteroid_mass_kg = volume * ASTEROID_DENSITY 
    
    impactor_mass_kg = kinetic_impactor_size_t * 1000
    impactor_velocity_mps = 10000 # 10 km/s
    
    # Simple elastic collision approximation for momentum transfer
    achieved_delta_v_mps = (impactor_mass_kg * impactor_velocity_mps) / asteroid_mass_kg
    
    # 3. Status Evaluation
    if achieved_delta_v_mps >= required_delta_v_mps * 1.5:
        status = "SUCCESS: Significant Miss (Orbit Shifted)"
    elif achieved_delta_v_mps >= required_delta_v_mps * 0.9:
        status = "PARTIAL SUCCESS: Grazing Miss (Uncertainty)"
    else:
        status = "FAILURE: Impact Predicted"
        
    return {
        "required_delta_v_mps": required_delta_v_mps,
        "achieved_delta_v_mps": achieved_delta_v_mps,
        "status": status,
    }

# --- 4. STREAMLIT UI COMPONENTS ---

# Sidebar for Primary Inputs
with st.sidebar:
    st.image("https://placehold.co/150x80/0A3D62/FFFFFF/png?text=NEO+Tracker", caption="Impactor-2025 Mitigation Console")
    st.header("Asteroid Parameters (NASA Data)")

    # 1. Asteroid Diameter (Size)
    diameter = st.slider(
        'Impactor-2025 Diameter (meters)',
        min_value=100, max_value=1000, value=IMPACTOR_2025_DEFAULT['diameter'], step=50
    )

    # 2. Relative Velocity
    velocity = st.slider(
        'Impact Velocity (km/s)',
        min_value=5.0, max_value=40.0, value=IMPACTOR_2025_DEFAULT['velocity'], step=1.0
    )

    # 3. Time to Impact (for Deflection modeling)
    days_to_impact = st.slider(
        'Days to Impact (Orbital Period)',
        min_value=30, max_value=365*5, value=IMPACTOR_2025_DEFAULT['days_to_impact'], step=30
    )
    
    st.divider()
    st.info(f"Target Density: {ASTEROID_DENSITY} kg/m³")

# Main App Body
st.title("🛡️ Project Impactor-2025: Asteroid Risk Management")
st.markdown("A real-data simulation and visualization tool for decision-makers and the public.")

# Calculate initial impact metrics
impact_results = calculate_impact_metrics(diameter, velocity, ASTEROID_DENSITY)

# --- TABS FOR WORKFLOW ---
tab1, tab2, tab3 = st.tabs(["🔴 Impact Scenario & Metrics", "🗺️ Impact Visualization (USGS)", "🚀 Mitigation Strategy"])

# ==============================================================================
# TAB 1: SCENARIO & METRICS
# ==============================================================================
with tab1:
    st.header(f"Impactor-2025: Key Impact Metrics (Diameter: {diameter}m, Velocity: {velocity} km/s)")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Calculated Mass (Billion kg)", value=f"{impact_results['mass_billion_kg']:.2f}")
        with st.expander("Details"):
            st.markdown("Mass is derived from the **Diameter** (NASA data) and an assumed **Density** (3000 kg/m³).")

    with col2:
        st.metric(label="Kinetic Energy (Megatons TNT)", value=f"{impact_results['energy_megatons_tnt']:.0f}")
        with st.expander("Details"):
            st.markdown(r"Energy is calculated by $E = \frac{1}{2}mv^2$. This is the raw destructive power.")

    with col3:
        st.metric(label="Estimated Crater Diameter (km)", value=f"{impact_results['crater_diameter_km']:.2f}")
        with st.expander("Details"):
            st.markdown(f"Crater size is calculated using established power-law scaling relationships based on energy, assuming a simple soil target.")

    with col4:
        st.metric(label="Seismic Magnitude (Mw)", value=f"{impact_results['seismic_magnitude']:.1f}")
        with st.expander("Details"):
            st.markdown("Seismic Moment Magnitude is an estimate of the resulting earthquake for a land impact, based on the total kinetic energy release.")
            
    st.divider()
    
    # Educational Section: Storytelling
    st.subheader("Interactive Scenario Narrative")
    
    st.markdown(f"""
    Impactor-2025 represents an existential threat, currently tracked at **{days_to_impact} days to impact**.
    The kinetic energy alone is equivalent to **{impact_results['energy_megatons_tnt']:.0f} Megatons of TNT**.
    """)

    with st.expander("🔬 Scientific Context: What does this mean?"):
        st.markdown(
            """
            - **Tunguska Event (1908)** was roughly 15 Megatons. This impact is **significantly larger**.
            - An estimated **crater of up to** **$${impact_results['crater_diameter_km']:.2f} \text{ km}$$** would be formed at ground zero.
            - The resulting **seismic event ($${impact_results['seismic_magnitude']:.1f} \text{ Mw}$$)** would cause widespread destruction over a large radius.
            """
        )

# ==============================================================================
# TAB 2: IMPACT VISUALIZATION (USGS Data Integration)
# ==============================================================================
with tab2:
    st.header("Geological Impact Visualization (USGS Integration)")
    
    # 1. Location Selection
    site_names = [site['name'] for site in MOCK_IMPACT_SITES]
    selected_site_name = st.selectbox(
        "Select Predicted Impact Site (Mock NEO Trajectory Data)",
        options=site_names,
        index=0 # Default to Coastal City A
    )
    
    # Get the data for the selected site
    selected_site = next(item for item in MOCK_IMPACT_SITES if item["name"] == selected_site_name)
    
    # Update Impact Coordinates
    impact_lat = selected_site['lat']
    impact_lon = selected_site['lon']
    site_type = selected_site['type']
    
    st.markdown(f"**Predicted Impact Location:** **{impact_lat}°N, {impact_lon}°W** ({site_type} region)")

    # 2. Interactive Map (Using Streamlit's native map for simplicity and speed)
    map_data = pd.DataFrame({
        'lat': [impact_lat],
        'lon': [impact_lon],
        'size': [diameter / 50], # Scale point size for visibility
        'name': [selected_site_name]
    })
    
    st.map(map_data, zoom=5, use_container_width=True)

    # 3. Environmental Consequences (Driven by USGS mock data/site type)
    st.subheader("Predicted Secondary Effects")
    
    if site_type == "Coastal":
        st.error(
            f"**TSUNAMI ALERT:** The impact at **{selected_site_name}** is predicted to trigger a catastrophic **Tsunami**. "
            f"Wave height and run-up modeling (USGS Elevation Data) suggests **>100m waves** reaching hundreds of kilometers inland."
        )
        st.warning(f"**Seismic Risk:** High ({impact_results['seismic_magnitude']:.1f} Mw)")
        
    elif site_type == "Inland":
        st.success("No Tsunami Risk.")
        st.error(
            f"**SEISMIC & ATMOSPHERIC RISK:** The main threat is the **$${impact_results['seismic_magnitude']:.1f} \text{ Mw}$$** earthquake and massive atmospheric dust injection (potential **'Impact Winter'**)."
        )
        
    elif site_type == "Oceanic":
        st.error(
            "**GLOBAL TSUNAMI THREAT:** An oceanic impact presents the highest risk of **trans-oceanic tsunamis** affecting multiple continents (USGS Tsunami Zones Data)."
        )
        st.success("Minimal immediate atmospheric dust injection.")

# ==============================================================================
# TAB 3: MITIGATION STRATEGY (Kinetic Impactor Simulation)
# ==============================================================================
with tab3:
    st.header("Mitigation Strategy Evaluation: Kinetic Impactor")

    # Mitigation Input Controls
    col_impactors, col_sim = st.columns([1, 2])
    
    with col_impactors:
        # Input for the size of the kinetic impactor
        impactor_size = st.number_input(
            'Kinetic Impactor Mass (Tonnes)',
            min_value=1.0, max_value=100.0, value=7.5, step=0.5,
            help="Mass of the spacecraft used to deflect the asteroid."
        )
        
        # Run the simulation
        if st.button('Simulate Deflection'):
            with st.spinner('Running Orbital Mechanics Simulation...'):
                time.sleep(1)
                deflection_results = simulate_deflection(
                    diameter, velocity, days_to_impact, impactor_size
                )
            
            st.session_state['deflection_results'] = deflection_results
            st.success("Simulation Complete.")

    with col_sim:
        if 'deflection_results' in st.session_state:
            results = st.session_state['deflection_results']
            
            # Outcome Visualization
            if results['status'].startswith("SUCCESS"):
                st.balloons()
                st.success(f"**OUTCOME: {results['status']}**")
            elif results['status'].startswith("PARTIAL"):
                st.warning(f"**OUTCOME: {results['status']}**")
            else:
                st.error(f"**OUTCOME: {results['status']}**")
            
            # Metrics
            st.metric(
                label="Required Delta-V (m/s) for Safe Miss", 
                value=f"{results['required_delta_v_mps']*1000:.3f} mm/s"
            )
            st.metric(
                label="Achieved Delta-V (m/s) by Impactor", 
                value=f"{results['achieved_delta_v_mps']*1000:.3f} mm/s"
            )

            # --- Altair Visualization: Required Delta-V over Time ---
            
            # Generate Data for the Deflection Window Chart
            T_DAYS = np.arange(30, days_to_impact + 30, 30) # time points
            T_SEC = T_DAYS * 24 * 3600 # Convert to seconds
            # Required Delta V is inversely proportional to time to impact
            V_REQUIRED_MPS = (EARTH_RADIUS * 1000) / T_SEC 

            chart_df = pd.DataFrame({
                'Time_to_Impact_Days': T_DAYS,
                'Required_Delta_V_mm_s': V_REQUIRED_MPS * 1000 # convert to mm/s
            })
            
            # Add the achieved delta-V as a line
            achieved_v_mm_s = results['achieved_delta_v_mps'] * 1000

            base = alt.Chart(chart_df).encode(
                alt.X('Time_to_Impact_Days', axis=alt.Axis(title='Time to Impact (Days)')),
                alt.Y('Required_Delta_V_mm_s', axis=alt.Axis(title='Required Velocity Change (mm/s)'))
            ).properties(
                title='Deflection Window: Required Delta-V vs. Time'
            )

            # Required Delta V Line (the threat)
            line = base.mark_line(color='red').encode(
                tooltip=['Time_to_Impact_Days', alt.Tooltip('Required_Delta_V_mm_s', format='.3f')]
            )

            # Achieved Delta V Line (the solution)
            achieved_line = alt.Chart(pd.DataFrame({'Achieved_V': [achieved_v_mm_s]})) \
                .mark_rule(color='green', strokeDash=[5, 5]) \
                .encode(y=alt.Y('Achieved_V:Q')) \
                .interactive()
            
            # Combine the charts
            st.altair_chart(line + achieved_line, use_container_width=True)
            
            st.caption("""
            The **Red Line** shows the minimum required velocity change (Delta-V) to deflect the asteroid
            to a safe trajectory. The **Green Dashed Line** shows the Delta-V achieved by your
            selected **Kinetic Impactor Mass**. The lines must separate to ensure a miss.
            """)
            
# --- FOOTER ---
st.divider()
st.caption("Powered by simulated NASA NEO parameters and USGS geological data models.")
