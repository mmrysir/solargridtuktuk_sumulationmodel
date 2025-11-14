# app.py
import traceback
import streamlit as st
from simulation import run_simulation
import numpy as np

st.set_page_config(page_title="Solar vs Electric TukTuk Simulator", layout="wide")
st.title("☀️ Solar TukTuk vs Electric TukTuk Simulator")

with st.sidebar:
    st.header("Simulation Settings")

    battery_capacity = st.slider("Battery Capacity (Wh)", 2000, 10000, 5000, 500)
    capacitor_capacity = st.slider("Capacitor Capacity (Wh)", 0, 2000, 500, 100)
    motor_power = st.slider("Motor Power (W)", 1000, 10000, 5000, 500)
    motor_efficiency = st.slider("Motor Efficiency", 0.70, 0.98, 0.85, 0.01)
    panel_area = st.slider("Panel Area (m²)", 0.5, 3.0, 1.5, 0.1)
    panel_efficiency = st.slider("Panel Efficiency", 0.10, 0.25, 0.20, 0.01)

    solar_power_output_factor = st.slider(
        "Solar Panel Power Output Multiplier",
        0.5, 2.0, 1.0, 0.05,
        help="Adjust this to simulate different solar panel power outputs."
    )

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
    years = st.slider("Years for Cost Projection", 1, 20, 10, 1)

st.info("Adjust parameters to see simulation results update automatically.")

try:
    results = run_simulation(
        battery_capacity=battery_capacity,
        capacitor_capacity=capacitor_capacity,
        motor_power=motor_power,
        motor_efficiency=motor_efficiency,
        panel_area=panel_area,
        panel_efficiency=panel_efficiency * solar_power_output_factor,
        trickle_charge=trickle_charge,
        solar_hours=solar_hours,
        distance_per_terrain=distance_per_terrain,
        grid_cost_per_kwh=grid_cost_per_kwh,
        initial_investment_solar=initial_investment_solar,
        initial_investment_electric=initial_investment_electric,
        annual_km=annual_km,
        base_kerb_weight=base_kerb_weight,
        panel_weight_per_m2=panel_weight_per_m2,
        cap_weight_per_Wh=cap_weight_per_Wh,
        years=years
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Energy Efficiency by Terrain")
        if "efficiency_plot" in results:
            st.pyplot(results['efficiency_plot'], clear_figure=True)
        else:
            st.write("No efficiency plot returned by simulation.")

    with col2:
        st.subheader("Estimated Range by Terrain")
        if "range_plot" in results:
            st.pyplot(results['range_plot'], clear_figure=True)
        else:
            st.write("No range plot returned by simulation.")

    st.subheader(f"Total Cost Projection over {years} Years")
    if "cost_plot" in results:
        st.pyplot(results['cost_plot'], clear_figure=True)

    st.subheader("Solar Power Output / Weather Impact")
    if "weather_impact_plot" in results:
        st.pyplot(results['weather_impact_plot'], clear_figure=True)
    elif "fig_weather" in results:
        st.pyplot(results['fig_weather'], clear_figure=True)
    else:
        st.write("No weather plot returned by simulation.")

    st.subheader("Carbon Emissions Comparison (kg CO2)")
    if "carbon_plot" in results:
        st.pyplot(results['carbon_plot'], clear_figure=True)
    elif "fig_carbon" in results:
        st.pyplot(results['fig_carbon'], clear_figure=True)

    st.subheader("Energy Efficiency Table (Wh/km)")
    if 'df_efficiency' in results:
        st.dataframe(results['df_efficiency'])
    elif 'df' in results:
        st.dataframe(results['df'])

    st.subheader("Range Table (km)")
    if 'df_range' in results:
        st.dataframe(results['df_range'])
    elif 'df' in results:
        st.dataframe(results['df'][['terrain', 'solar_range_km', 'electric_range_km']])

    st.subheader("Cost Summary Table")
    if 'df_cost' in results:
        st.dataframe(results['df_cost'])

    # ----- FIXED METRICS -----
    if 'total_solar_power_generated_Wh' in results:
        st.subheader("Total Solar Power Generated (Wh)")
        st.metric(label="Total Solar Power", value=f"{results['total_solar_power_generated_Wh']:.2f}")

    if 'carbon_emissions_kg_grid' in results:
        st.subheader("Carbon Emissions from Grid (kg CO2)")
        st.metric(label="Grid Emissions", value=f"{results['carbon_emissions_kg_grid']:.2f}")

    if 'carbon_emissions_kg_solar' in results:
        st.subheader("Carbon Emissions from Solar (kg CO2)")
        st.metric(label="Solar Emissions", value=f"{results['carbon_emissions_kg_solar']:.2f}")

except Exception:
    st.error("Simulation failed — see details below.")
    st.text(traceback.format_exc())
