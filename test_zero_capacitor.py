#!/usr/bin/env python
# coding: utf-8

"""
Test to verify that when capacitor is 0, solar and grid have matching efficiency
"""

from simulation import run_simulation
import matplotlib.pyplot as plt

print("Testing: When capacitor_capacity = 0, solar and grid should match\n")

# Run simulation with zero capacitor
results = run_simulation(
    capacitor_capacity=0,  # Zero capacitor
    battery_capacity=5000,
    motor_power=5000,
    motor_efficiency=0.85,
    panel_area=1.5,
    panel_efficiency=0.2,
    power_output_multiplier=1.0,
    distance_per_terrain=10,
    weather_type="Clear"
)

# Check the efficiency DataFrame
df = results['df_efficiency']
print("Energy Efficiency by Terrain (capacitor = 0):")
print(df)
print()

# Verify they match
all_match = True
for idx, row in df.iterrows():
    solar_eff = row['SolarTukTuk (Wh/km)']
    grid_eff = row['Grid TukTuk (Wh/km)']
    terrain = row['Terrain']
    
    # Allow tiny floating point differences
    if abs(solar_eff - grid_eff) > 0.01:
        print(f"❌ MISMATCH on {terrain}: Solar={solar_eff:.2f}, Grid={grid_eff:.2f}")
        all_match = False
    else:
        print(f"✓ {terrain}: Solar={solar_eff:.2f}, Grid={grid_eff:.2f} - MATCH!")

print()
if all_match:
    print("✅ SUCCESS: All terrains match when capacitor is 0!")
else:
    print("❌ FAILURE: Some terrains don't match")

# Show the efficiency plot to visually verify
results['efficiency_plot'].suptitle('Efficiency Plot (Capacitor = 0)', fontsize=16, fontweight='bold', y=1.02)
plt.show()
