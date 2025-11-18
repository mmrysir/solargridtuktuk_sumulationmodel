import traceback
import streamlit as st
from simulation import run_simulation
import numpy as np

st.set_page_config(page_title="Solar vs Electric TukTuk Simulator", layout="wide")
st.title("☀️ Solar TukTuk vs Electric TukTuk Simulator")

with st.sidebar:
    st.header("Simulation Settings")
    
    st.subheader("Battery & Motor")
    battery_capacity = st.slider("Battery Capacity (Wh)", 2000, 10000, 5000, 500)
    capacitor_capacity = st.slider("Capacitor Capacity (Wh)", 0, 2000, 500, 100)
    motor_power = st.slider("Motor Power (W)", 1000, 10000, 5000, 500)
    motor_efficiency = st.slider("Motor Efficiency", 0.70, 0.98, 0.85, 0.01)
    
    st.subheader("Solar Panel")
    panel_area = st.slider("Panel Area (m²)", 0.5, 3.0, 1.5, 0.1)
    panel_efficiency = st.slider("Panel Efficiency", 0.10, 0.25, 0.20, 0.01)
    solar_power_output_factor = st.slider(
        "Solar Panel Power Output Multiplier",
        0.5, 2.0, 1.0, 0.05,
        help="Adjust this to simulate different solar panel power outputs."
    )
    
    st.subheader("Operational Settings")
    trickle_charge = st.checkbox("Trickle Charge (Solar)", value=True)
    solar_hours = st.slider("Solar Hours", 0, 12, 6, 1)
    distance_per_terrain = st.slider("Distance per Terrain (km)", 1, 50, 10, 1)
    
    st.subheader("Cost Parameters")
    grid_cost_per_kwh = st.slider("Grid Cost ($/kWh)", 0.05, 0.50, 0.20, 0.01)
    initial_investment_solar = st.slider("Solar Initial Investment ($)", 5000, 20000, 9000, 500)
    initial_investment_electric = st.slider("Electric Initial Investment ($)", 3000, 15000, 6000, 500)
    annual_km = st.slider("Annual Distance (km)", 5000, 50000, 20000, 1000)
    
    st.subheader("Weight & Physical")
    base_kerb_weight = st.slider("Base Kerb Weight (kg)", 300, 700, 400, 10)
    panel_weight_per_m2 = st.slider("Panel Weight (kg/m²)", 8.0, 20.0, 12.5, 0.5)
    cap_weight_per_Wh = st.slider("Capacitor Weight (kg/Wh)", 0.02, 0.08, 0.04, 0.005)
    
    st.subheader("Projection Period")
    years = st.slider("Years for Cost Projection", 1, 20, 10, 1)

st.info("✨ Adjust parameters to see simulation results update automatically.")

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

    # Section 1: Weather Impact
    st.markdown("---")
    st.header("🌤️ Solar Power & Weather Analysis")
    
    if "weather_impact_plot" in results:
        st.pyplot(results['weather_impact_plot'])
    else:
        st.warning("Weather impact plot not available")

    # Section 2: Supercapacitor vs Battery Efficiency
    st.markdown("---")
    st.header("🔋 Supercapacitor vs Battery Efficiency")
    
    if "supercap_vs_battery_plot" in results:
        st.pyplot(results['supercap_vs_battery_plot'])
        st.info("💡 **Left Panel**: Energy Source Distribution shows Solar TukTuk uses hybrid battery (80%) + supercapacitor (20%) system for each terrain type vs Grid TukTuk's 100% battery reliance.\n\n**Right Panel**: Supercapacitors maintain >95% efficiency even after 10,000 cycles, while batteries degrade from 95% to 68%. This demonstrates massive longevity advantages! ⚡")
    else:
        st.warning("Supercapacitor plot not available")

    # Section 3: Battery Degradation & Replacement Costs
    st.markdown("---")
    st.header("📉 Battery Degradation & Replacement Costs")
    
    if "degradation_plot" in results:
        st.pyplot(results['degradation_plot'])
        st.info("💚 **Left Panel**: Solar TukTuk reaches 80% capacity at ~7.4 years thanks to supercapacitor reducing stress, while Grid TukTuk reaches it at ~3.7 years.\n\n**Right Panel**: Accounting for battery replacement costs ($2,000), Solar TukTuk saves significantly by extending battery life! 🌱")
    else:
        st.warning("Battery degradation plot not available")

    # Section 4: Performance Metrics
    st.markdown("---")
    st.header("📊 Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Energy Efficiency by Terrain")
        if "efficiency_plot" in results:
            st.pyplot(results['efficiency_plot'])
        else:
            st.warning("Efficiency plot not available")
    
    with col2:
        st.subheader("Estimated Range by Terrain")
        if "range_plot" in results:
            st.pyplot(results['range_plot'])
        else:
            st.warning("Range plot not available")

    # Section 5: Cost Analysis
    st.markdown("---")
    st.header("💰 Cost Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Total Cost Projection ({years} Years)")
        if "cost_plot" in results:
            st.pyplot(results['cost_plot'])
        else:
            st.warning("Cost plot not available")
    
    with col2:
        st.subheader("Carbon Emissions Comparison")
        if "carbon_plot" in results:
            st.pyplot(results['carbon_plot'])
        else:
            st.warning("Carbon emissions plot not available")

    # Section 6: Detailed Data Tables
    st.markdown("---")
    st.header("📋 Detailed Data Tables")
    
    tab1, tab2, tab3 = st.tabs(["Energy Efficiency", "Range", "Cost"])
    
    with tab1:
        st.subheader("Energy Efficiency (Wh/km)")
        if 'df_efficiency' in results:
            st.dataframe(results['df_efficiency'])
        else:
            st.info("Efficiency data not available")
    
    with tab2:
        st.subheader("Range by Terrain (km)")
        if 'df_range' in results:
            st.dataframe(results['df_range'])
        else:
            st.info("Range data not available")
    
    with tab3:
        st.subheader("Cost Summary")
        if 'df_cost' in results:
            st.dataframe(results['df_cost'])
        else:
            st.info("Cost data not available")

    # Section 7: Key Metrics
    st.markdown("---")
    st.header("⚡ Key Performance Metrics")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        if 'total_solar_power_generated_Wh' in results:
            st.metric(
                label="Total Solar Power Generated",
                value=f"{results['total_solar_power_generated_Wh']:.2f} Wh"
            )
    
    with metric_col2:
        if 'carbon_emissions_kg_grid' in results:
            st.metric(
                label="Grid Emissions (Annual)",
                value=f"{results['carbon_emissions_kg_grid']:.2f} kg CO₂"
            )
    
    with metric_col3:
        if 'carbon_emissions_kg_solar' in results:
            st.metric(
                label="Solar Emissions (Annual)",
                value=f"{results['carbon_emissions_kg_solar']:.2f} kg CO₂"
            )

    # Section 8: Summary Statistics
    st.markdown("---")
    st.header("📈 Summary Statistics")
    
    if 'carbon_emissions_kg_grid' in results and 'carbon_emissions_kg_solar' in results:
        emissions_diff = results['carbon_emissions_kg_grid'] - results['carbon_emissions_kg_solar']
        emissions_reduction = (emissions_diff / results['carbon_emissions_kg_grid'] * 100) if results['carbon_emissions_kg_grid'] > 0 else 0
        
        st.success(f"🌱 **Solar TukTuk reduces emissions by {emissions_reduction:.1f}% annually**")
        st.info(f"Annual CO₂ savings: **{emissions_diff:.2f} kg CO₂**")

except Exception as e:
    st.error("❌ Simulation failed — see details below.")
    st.text(traceback.format_exc())