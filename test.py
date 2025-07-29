import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from simulation import run_simulation  # Your simulation function

st.title("Tuktuk Solar vs Electric Simulation Dashboard")

# Sidebar for user input
battery_capacity = st.sidebar.slider("Battery Capacity (Wh)", 2000, 10000, 5000, 500)
capacitor_capacity = st.sidebar.slider("Capacitor Capacity (Wh)", 0, 2000, 500, 100)
motor_power = st.sidebar.slider("Motor Power (W)", 1000, 10000, 5000, 500)
motor_efficiency = st.sidebar.slider("Motor Efficiency", 0.7, 0.98, 0.85, 0.01)
panel_area = st.sidebar.slider("Panel Area (m²)", 0.5, 3.0, 1.5, 0.1)
panel_efficiency = st.sidebar.slider("Panel Efficiency", 0.1, 0.25, 0.2, 0.01)
trickle_charge = st.sidebar.checkbox("Trickle Charge (Solar)", value=True)
solar_hours = st.sidebar.slider("Solar Hours", 0, 12, 6, 1)
distance_per_terrain = st.sidebar.slider("Distance per Terrain (km)", 1, 50, 10, 1)
grid_cost_per_kwh = st.sidebar.slider("Grid Cost ($/kWh)", 0.05, 0.50, 0.20, 0.01)
initial_investment_solar = st.sidebar.slider("Solar Initial Investment ($)", 5000, 20000, 9000, 500)
initial_investment_electric = st.sidebar.slider("Electric Initial Investment ($)", 3000, 15000, 6000, 500)
annual_km = st.sidebar.slider("Annual Distance (km)", 5000, 50000, 20000, 1000)
base_kerb_weight = st.sidebar.slider("Base Kerb Weight (kg)", 300, 700, 400, 10)
panel_weight_per_m2 = st.sidebar.slider("Panel Weight (kg/m²)", 8.0, 20.0, 12.5, 0.5)
cap_weight_per_Wh = st.sidebar.slider("Capacitor Weight (kg/Wh)", 0.02, 0.08, 0.04, 0.005)

if st.button("Run Simulation"):
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
    # Show plots
    st.subheader("Energy Efficiency by Terrain")
    st.pyplot(results['efficiency_plot'])
    st.subheader("Estimated Range by Terrain")
    st.pyplot(results['range_plot'])
    st.subheader("Total Cost Over 1 Year")
    st.pyplot(results['cost_plot'])
    # Show DataFrames
    st.subheader("Energy Efficiency Table")
    st.dataframe(results['df_efficiency'])
    st.subheader("Range Table")
    st.dataframe(results['df_range'])
    st.subheader("Running Cost Table")
    st.dataframe(results['df_cost'])
