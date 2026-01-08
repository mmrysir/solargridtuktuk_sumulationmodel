import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mock data based on defaults in simulation.py
years = 10
initial_investment_solar = 9000
initial_investment_electric = 6000
annual_km = 20000
grid_cost_per_kwh = 0.20

# Mock DataFrame for efficiency
# Approx values from simulation logic
df = pd.DataFrame({
    'grid_eff_Wh_per_km': [35, 40, 45, 50, 20] # Average around 38
})

# Calculate costs
df['grid_cost_per_km'] = df['grid_eff_Wh_per_km'] / 1000 * grid_cost_per_kwh
avg_grid_cost_per_km = df['grid_cost_per_km'].mean()

years_arr = np.arange(0, years + 1)
solar_total_cost = np.full_like(years_arr, initial_investment_solar, dtype=float)
grid_total_cost = initial_investment_electric + (avg_grid_cost_per_km * annual_km * years_arr)

# Plotting code from simulation.py
fig_cost, ax_cost = plt.subplots(figsize=(12, 6))
ax_cost.plot(years_arr, solar_total_cost, label='Solar TukTuk (No Grid Cost)', linewidth=2.5, color='#1f77b4', marker='o', markersize=6)
ax_cost.plot(years_arr, grid_total_cost, label='Grid TukTuk (Energy Cost Accumulation)', linewidth=2.5, color='#ff7f0e', marker='s', markersize=6)

# The problematic fill_between
ax_cost.fill_between(years_arr, solar_total_cost, grid_total_cost, color='#ff7f0e', alpha=0.1, label="Cost Savings with Solar")

ax_cost.set_xlabel('Years of Operation', fontsize=11)
ax_cost.set_ylabel('Total Cost (USD)', fontsize=11)
ax_cost.set_title(f'Total Cost Projection over {years} Years', fontsize=13, fontweight='bold')
ax_cost.set_ylim(bottom=0)
ax_cost.legend(fontsize=10)
ax_cost.grid(True, linestyle='--', alpha=0.7)

# The problematic text annotations
ax_cost.text(years_arr[-1], solar_total_cost[-1], f"${solar_total_cost[-1]:,.0f}", va='bottom', ha='right', color='#1f77b4', fontweight='bold')
ax_cost.text(years_arr[-1], grid_total_cost[-1], f"${grid_total_cost[-1]:,.0f}", va='top', ha='right', color='#ff7f0e', fontweight='bold')

plt.tight_layout()
plt.savefig('cost_plot_repro.png')
print("Plot saved to cost_plot_repro.png")
