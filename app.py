import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from simulation import run_simulation

# Improved CSS for scrollable sidebar with thicker scrollbar
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        overflow-y: auto;
        max-height: 100vh;
    }

    section[data-testid="stSidebar"]::-webkit-scrollbar {
        width: 14px;
    }

    section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background-color: #888;
        border-radius: 6px;
    }

    section[data-testid="stSidebar"]::-webkit-scrollbar-track {
        background-color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Solar Tuktuk v Grid TuktukSimulation Dashboard")

# Sidebar for simulation parameters
with st.sidebar:
    st.header("Simulation Parameters")

    battery_capacity = st.slider("Battery Capacity (Wh)", 2000, 10000, 5000, 500)
    capacitor_capacity = st.slider("Capacitor Capacity (Wh)", 0, 2000, 500, 100)
    motor_power = st.slider("Motor Power (W)", 1000, 10000, 5000, 500)
    motor_efficiency = st.slider("Motor Efficiency", 0.7, 0.98, 0.85, 0.01)
    panel_area = st.slider("Panel Area (m²)", 0.5, 3.0, 1.5, 0.1)
    panel_efficiency = st.slider("Panel Efficiency", 0.1, 0.25, 0.2, 0.01)
    trickle_charge = st.checkbox("Trickle Charge (Solar)", value=True)
    solar_hours = st.slider("Solar Hours", 0, 12, 6, 1)
    distance_per_terrain = st.slider("Distance per Terrain (km)", 1, 50, 10, 1)
    grid_cost_per_kwh = st.slider("Grid Cost ($/kWh)", 0.05, 0.50, 0.20, 0.01)
    initial_investment_solar = st.slider("Solar Initial Investment ($)", 5000, 20000, 9000, 500)
    initial_investment_electric = st.slider("Electric Initial Investment ($)", 3000, 15000, 6000, 500)
    annual_km = st.slider("Annual Distance (km)", 5000, 50000, 20000, 1000)
    base_kerb_weight = st.slider("Base Kerb Weight (kg)", 300, 700, 400, 10)
    panel_weight_per_m2 = st.slider("Panel Weight (kg/m²)", 8.0, 20.0, 12.5, 0.5)
    cap_weight_per_Wh = st.slider("Capacitor Weight (kg/Wh)", 0.02, 0.08, 0.04, 0.005)

    st.info("Adjust the parameters to simulate and compare energy performance.")

# Run simulation
results = run_simulation(
    battery_capacity=battery_capacity,
    capacitor_capacity=capacitor_capacity,
    motor_power=motor_power,
    motor_efficiency=motor_efficiency,
    panel_area=panel_area,
    panel_efficiency=panel_efficiency,
    trickle_charge=trickle_charge,
    solar_hours=solar_hours,
    distance_per_terrain=distance_per_terrain,
    grid_cost_per_kwh=grid_cost_per_kwh,
    initial_investment_solar=initial_investment_solar,
    initial_investment_electric=initial_investment_electric,
    annual_km=annual_km,
    base_kerb_weight=base_kerb_weight,
    panel_weight_per_m2=panel_weight_per_m2,
    cap_weight_per_Wh=cap_weight_per_Wh
)

# Display plots
st.subheader("Energy Efficiency by Terrain")
st.pyplot(results['efficiency_plot'])

st.subheader("Estimated Range by Terrain")
st.pyplot(results['range_plot'])

st.subheader("Total Cost Over 1 Year")
st.pyplot(results['cost_plot'])

# Display data
st.subheader("Energy Efficiency Table")
st.dataframe(results['df_efficiency'])

st.subheader("Range Table")
st.dataframe(results['df_range'])

st.subheader("Running Cost Table")
st.dataframe(results['df_cost'])

# Download buttons
st.download_button("Download Efficiency Data", results['df_efficiency'].to_csv(index=False), "efficiency.csv")
st.download_button("Download Range Data", results['df_range'].to_csv(index=False), "range.csv")
st.download_button("Download Cost Data", results['df_cost'].to_csv(index=False), "cost.csv")
