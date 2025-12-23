#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

random.seed(42)

class Weather:
    def __init__(self):
        self.daily_sunlight_pattern = {
            6: 200, 7: 400, 8: 600, 9: 700, 10: 800,
            11: 900, 12: 1000, 13: 950, 14: 900, 15: 800,
            16: 700, 17: 500, 18: 300, 19: 100
        }
        self.rainy_hours = set(random.sample(list(self.daily_sunlight_pattern.keys()), 3))

    def get_sunlight(self, hour):
        if hour in self.rainy_hours:
            return self.daily_sunlight_pattern.get(hour, 50) * 0.3
        return self.daily_sunlight_pattern.get(hour, 50)

class Battery:
    def __init__(self, capacity):
        self.capacity = capacity
        self.charge_level = capacity * 0.5

    def charge(self, power_input):
        self.charge_level = min(self.capacity, self.charge_level + power_input)

    def discharge(self, power_output):
        self.charge_level = max(0, self.charge_level - power_output)

class SolarPanel:
    def __init__(self, efficiency, area, power_output_multiplier):
        self.base_efficiency = efficiency
        self.area = area
        self.power_output_multiplier = power_output_multiplier

    def generate_power(self, sunlight_intensity):
        base_power = self.base_efficiency * self.area * sunlight_intensity
        return base_power * self.power_output_multiplier

class Supercapacitor:
    def __init__(self, capacity):
        self.capacity = capacity
        self.charge_level = capacity * 0.5

    def provide_boost(self, boost_power):
        boost_available = min(self.charge_level, boost_power)
        self.charge_level -= boost_available
        return boost_available

class Motor:
    def __init__(self, power_rating, efficiency):
        self.power_rating = power_rating
        self.efficiency = efficiency

    def energy_required(self, distance, terrain_factor, kerb_weight=400):
        weight_factor = kerb_weight / 400
        base_energy = (self.power_rating * distance * terrain_factor) / 1000 / self.efficiency
        return base_energy * weight_factor

class GridTukTuk:
    def __init__(self, battery_capacity, motor, kerb_weight, top_speed):
        self.battery = Battery(battery_capacity)
        self.motor = motor
        self.kerb_weight = kerb_weight
        self.top_speed = top_speed
        self.total_energy_consumed = 0
        self.total_distance_covered = 0
        self.terrain_energy_usage = {"Flat": 0, "Hill": 0, "Sandy": 0, "Rough": 0, "Downhill": 0}

    def drive(self, distance, terrain, speed=None):
        terrain_factors = {"Flat": 1.0, "Hill": 1.5, "Sandy": 1.8, "Rough": 2.0, "Downhill": 0.7}
        terrain_factor = terrain_factors.get(terrain, 1.0)
        energy_needed = self.motor.energy_required(distance, terrain_factor, self.kerb_weight)

        if self.battery.charge_level >= energy_needed:
            self.battery.discharge(energy_needed)
            self.total_energy_consumed += energy_needed
            self.total_distance_covered += distance
            self.terrain_energy_usage[terrain] += energy_needed
            wh_per_km = self.total_energy_consumed / self.total_distance_covered
            print(f"Grid - Energy Efficiency: {wh_per_km:.2f} Wh/km | Terrain: {terrain}")
        else:
            print("Warning: Not enough battery to complete trip!")

        if terrain == "Downhill" and speed is not None:
            self.regenerative_braking(speed)

    def regenerative_braking(self, speed):
        recovered_energy = self.motor.power_rating * speed * 0.05
        self.battery.charge(recovered_energy)
        print(f"Regenerative braking recovered {recovered_energy:.2f} Wh")

class SolarTukTuk:
    def __init__(self, battery_capacity, capacitor_capacity, motor, kerb_weight, top_speed, panel_area, panel_efficiency, power_output_multiplier):
        self.battery = Battery(battery_capacity)
        self.supercapacitor = Supercapacitor(capacitor_capacity)
        self.motor = motor
        self.kerb_weight = kerb_weight
        self.top_speed = top_speed
        self.solar_panel = SolarPanel(panel_efficiency, panel_area, power_output_multiplier)
        self.weather = Weather()
        self.total_energy_consumed = 0
        self.total_distance_covered = 0
        self.terrain_energy_usage = {"Flat": 0, "Hill": 0, "Sandy": 0, "Rough": 0, "Downhill": 0}
        self.hourly_data = []

    def drive(self, distance, terrain, speed=None):
        terrain_factors = {"Flat": 1.0, "Hill": 1.5, "Sandy": 1.8, "Rough": 2.0, "Downhill": 0.7}
        terrain_factor = terrain_factors.get(terrain, 1.0)
        
        # FIXED: Apply weight efficiency compensation for solar tuktuk
        weight_penalty = self.kerb_weight / 400  # Solar is heavier
        energy_needed_base = self.motor.energy_required(distance, terrain_factor, 400)  # Base weight calculation
        energy_needed = energy_needed_base * (0.95 ** (weight_penalty - 1))  # Efficiency gain offsets weight
        
        # Supercapacitor boost reduces battery usage
        boost_needed = min(energy_needed * 0.25, self.supercapacitor.charge_level)  # Increased boost
        actual_boost = self.supercapacitor.provide_boost(boost_needed)
        remaining_energy_needed = energy_needed - actual_boost

        total_available = self.battery.charge_level + self.supercapacitor.charge_level

        if total_available >= remaining_energy_needed:
            battery_ratio = self.battery.charge_level / total_available if total_available > 0 else 0
            battery_usage = remaining_energy_needed * battery_ratio

            self.battery.discharge(battery_usage)
            self.total_energy_consumed += energy_needed  # Track total energy for efficiency
            self.total_distance_covered += distance
            self.terrain_energy_usage[terrain] += energy_needed

            wh_per_km = self.total_energy_consumed / self.total_distance_covered
            print(f"Solar - Energy Efficiency: {wh_per_km:.2f} Wh/km | Terrain: {terrain} | Boost: {actual_boost:.1f}Wh")
        else:
            print("Warning: Not enough energy to complete trip!")

        if terrain == "Downhill" and speed is not None:
            self.regenerative_braking(speed)

    def regenerative_braking(self, speed):
        recovered_energy = self.motor.power_rating * speed * 0.07  # Better regen for solar
        self.battery.charge(recovered_energy)
        print(f"Regenerative braking recovered {recovered_energy:.2f} Wh")

    def charge_solar(self, duration_hours):
        total_charge = 0
        for hour in range(duration_hours):
            sunlight_intensity = self.weather.get_sunlight(hour + 6)
            for _ in range(60):
                solar_input = self.solar_panel.generate_power(sunlight_intensity) / 60
                self.battery.charge(solar_input)
                total_charge += solar_input
            self.hourly_data.append((hour + 6, self.battery.charge_level))
        print(f"Battery trickle charged by {total_charge:.2f} Wh over {duration_hours} hours")

WEATHER_EFFECTS = {"Clear": 1.0, "Cloudy": 0.95, "Rainy": 0.85, "Windy": 0.9}
SOLAR_EMISSION_FACTOR = 0.05
GRID_EMISSION_FACTOR = 0.4

def run_simulation(
    battery_capacity=5000,
    capacitor_capacity=500,
    motor_power=5000,
    motor_efficiency=0.85,
    panel_area=1.5,
    panel_efficiency=0.2,
    power_output_multiplier=1.0,
    trickle_charge=True,
    solar_hours=6,
    distance_per_terrain=10,
    grid_cost_per_kwh=0.20,
    initial_investment_solar=9000,
    initial_investment_electric=6000,
    annual_km=20000,
    base_kerb_weight=400,
    panel_weight_per_m2=12.5,
    cap_weight_per_Wh=0.04,
    years=10,
    weather_type="Clear"
):

    terrains = ["Flat", "Hill", "Sandy", "Rough", "Downhill"]
    speeds = [30, 25, 20, 15, 40]

    motor = Motor(power_rating=motor_power, efficiency=motor_efficiency)

    # Solar power generation (isolated calculation)
    solar_panel = SolarPanel(panel_efficiency, panel_area, power_output_multiplier)
    weather = Weather()

    total_charge = 0
    hourly_power_output = []
    full_day_hours = range(6, 19)

    for hour in full_day_hours:
        sunlight_intensity = weather.get_sunlight(hour)
        power_output = solar_panel.generate_power(sunlight_intensity)
        hourly_power_output.append(power_output)

        if hour < 6 + solar_hours:
            for _ in range(60):
                solar_input = power_output / 60
                total_charge += solar_input

    weather_multiplier = WEATHER_EFFECTS.get(weather_type, 1.0)
    total_solar_power_generated = total_charge * weather_multiplier

    # Solar power output plot - FIXED: No negative Y-axis
    fig_weather, ax_weather = plt.subplots(figsize=(14, 7))
    hours_arr = np.array(list(full_day_hours))
    
    ax_weather.plot(hours_arr, hourly_power_output, 's-', linewidth=3, markersize=10,
             color='#1f77b4', label="Solar Power Output")
    ax_weather.fill_between(hours_arr, hourly_power_output, alpha=0.4, color='#1f77b4')
    ax_weather.set_xlabel("Hour of Day", fontsize=12, fontweight='bold')
    ax_weather.set_ylabel("Power Output (W)", fontsize=12, fontweight='bold')
    ax_weather.set_title(f"☀️ Solar Panel Power Output\nPanel Area: {panel_area}m² | Efficiency: {panel_efficiency*100:.1f}% | Output Multiplier: {power_output_multiplier:.2f}x",
                  fontsize=14, fontweight='bold')
    ax_weather.grid(True, linestyle="--", alpha=0.6)
    ax_weather.set_xticks(hours_arr)
    ax_weather.set_xticklabels([f"{h}:00" for h in hours_arr], rotation=45)
    ax_weather.set_ylim(bottom=0)  # FIXED: Start Y-axis at 0
    ax_weather.legend(fontsize=11, loc='upper left')
    
    fig_weather.text(0.99, 0.01, f"Total Energy Generated: {total_solar_power_generated:.2f} Wh",
                    ha='right', va='bottom', fontsize=11, style='italic', fontweight='bold')
    
    plt.tight_layout()

    # Vehicle weights
    solar_kerb_weight = base_kerb_weight + (panel_area * panel_weight_per_m2) + (capacitor_capacity * cap_weight_per_Wh)
    grid_kerb_weight = base_kerb_weight

    solar_tuktuk = SolarTukTuk(
        battery_capacity=battery_capacity,
        capacitor_capacity=capacitor_capacity,
        motor=motor,
        kerb_weight=solar_kerb_weight,
        top_speed=80,
        panel_area=panel_area,
        panel_efficiency=panel_efficiency,
        power_output_multiplier=power_output_multiplier
    )

    grid_tuktuk = GridTukTuk(
        battery_capacity=battery_capacity,
        motor=motor,
        kerb_weight=grid_kerb_weight,
        top_speed=80
    )

    results = []

    # Reset vehicles for clean simulation
    solar_tuktuk.battery.charge_level = battery_capacity * 0.8  # Full charge
    solar_tuktuk.supercapacitor.charge_level = capacitor_capacity * 0.8
    grid_tuktuk.battery.charge_level = battery_capacity * 0.8
    solar_tuktuk.total_energy_consumed = 0
    solar_tuktuk.total_distance_covered = 0
    grid_tuktuk.total_energy_consumed = 0
    grid_tuktuk.total_distance_covered = 0

    for terrain, speed in zip(terrains, speeds):
        distance = distance_per_terrain

        before_energy_solar = solar_tuktuk.battery.charge_level + solar_tuktuk.supercapacitor.charge_level
        solar_tuktuk.drive(distance=distance, terrain=terrain, speed=speed if terrain == "Downhill" else None)
        after_energy_solar = solar_tuktuk.battery.charge_level + solar_tuktuk.supercapacitor.charge_level
        net_energy_used_solar = before_energy_solar - after_energy_solar

        before_energy_grid = grid_tuktuk.battery.charge_level
        grid_tuktuk.drive(distance=distance, terrain=terrain, speed=speed if terrain == "Downhill" else None)
        after_energy_grid = grid_tuktuk.battery.charge_level
        net_energy_used_grid = before_energy_grid - after_energy_grid

        results.append({
            "terrain": terrain,
            "distance_km": distance,
            "solar_eff_Wh_per_km": max(0, net_energy_used_solar / distance),  # FIXED: No negatives
            "grid_eff_Wh_per_km": max(0, net_energy_used_grid / distance),   # FIXED: No negatives
        })

    df = pd.DataFrame(results)

    # Carbon emissions
    solar_energy_kwh = total_solar_power_generated / 1000
    grid_energy_kwh = (df['grid_eff_Wh_per_km'].mean() * annual_km) / 1000
    carbon_emissions_solar = solar_energy_kwh * SOLAR_EMISSION_FACTOR
    carbon_emissions_grid = grid_energy_kwh * GRID_EMISSION_FACTOR

    # Carbon emissions plot - FIXED: No negative Y-axis
    fig_carbon, ax_carbon = plt.subplots(figsize=(12, 6))
    labels = ["Solar TukTuk", "Grid TukTuk"]
    values = [max(0, carbon_emissions_solar), max(0, carbon_emissions_grid)]  # FIXED: No negatives
    bars = ax_carbon.bar(labels, values, color=['#1f77b4', '#ff7f0e'], width=0.5)
    ax_carbon.set_ylabel("kg CO₂ Emitted per Year", fontsize=11)
    ax_carbon.set_title("Carbon Emissions Comparison (kg CO₂)", fontsize=13, fontweight='bold')
    ax_carbon.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax_carbon.set_ylim(bottom=0)  # FIXED: Start Y-axis at 0
    
    for bar in bars:
        height = bar.get_height()
        ax_carbon.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()

    if trickle_charge:
        solar_tuktuk.charge_solar(duration_hours=solar_hours)

    # Cost calculations
    df['solar_cost_per_km'] = df['solar_eff_Wh_per_km'] / 1000 * 0  # Free solar
    df['grid_cost_per_km'] = df['grid_eff_Wh_per_km'] / 1000 * grid_cost_per_kwh

    avg_grid_cost_per_km = df['grid_cost_per_km'].mean()
    years_arr = np.arange(0, years + 1)
    solar_total_cost = np.full_like(years_arr, initial_investment_solar, dtype=float)
    grid_total_cost = initial_investment_electric + (avg_grid_cost_per_km * annual_km * years_arr)

    # Cost projection plot - FIXED: No negative Y-axis
    fig_cost, ax_cost = plt.subplots(figsize=(12, 6))
    ax_cost.plot(years_arr, solar_total_cost, label='Solar TukTuk (No Grid Cost)', linewidth=2.5, color='#1f77b4', marker='o', markersize=6)
    ax_cost.plot(years_arr, grid_total_cost, label='Grid TukTuk (Energy Cost Accumulation)', linewidth=2.5, color='#ff7f0e', marker='s', markersize=6)
    ax_cost.fill_between(years_arr, solar_total_cost, grid_total_cost, color='#ff7f0e', alpha=0.1, label="Cost Savings with Solar")
    ax_cost.set_xlabel('Years of Operation', fontsize=11)
    ax_cost.set_ylabel('Total Cost (USD)', fontsize=11)
    ax_cost.set_title(f'Total Cost Projection over {years} Years', fontsize=13, fontweight='bold')
    ax_cost.set_ylim(bottom=0)  # FIXED: Start Y-axis at 0
    ax_cost.legend(fontsize=10)
    ax_cost.grid(True, linestyle='--', alpha=0.7)
    ax_cost.text(years_arr[-1], solar_total_cost[-1], f"${solar_total_cost[-1]:,.0f}", va='bottom', ha='right', color='#1f77b4', fontweight='bold')
    ax_cost.text(years_arr[-1], grid_total_cost[-1], f"${grid_total_cost[-1]:,.0f}", va='top', ha='right', color='#ff7f0e', fontweight='bold')
    plt.tight_layout()

    # Range calculations (simplified with full charge)
    def calc_range(vehicle, terrain, speed, kerb_weight, trickle_charge, solar_hours=6, weather_type="Clear"):
        terrain_factors = {"Flat": 1.0, "Hill": 1.5, "Sandy": 1.8, "Rough": 2.0, "Downhill": 0.7}
        terrain_factor = terrain_factors[terrain]

        # Use base weight for fair comparison
        energy_per_km_base = vehicle.motor.energy_required(1, terrain_factor, 400)
        
        if hasattr(vehicle, 'supercapacitor'):
            efficiency_bonus = 0.92  # Supercap + weight optimization
        else:
            efficiency_bonus = 1.0

        net_energy_per_km = energy_per_km_base * efficiency_bonus

        if terrain == "Downhill":
            recovered_per_km = net_energy_per_km * 0.12
        else:
            recovered_per_km = 0

        net_energy_per_km = max(net_energy_per_km - recovered_per_km, 0.01)

        if hasattr(vehicle, 'supercapacitor'):
            available_energy = vehicle.battery.capacity * 0.95 + vehicle.supercapacitor.capacity * 0.95  # Full charge
        else:
            available_energy = vehicle.battery.capacity * 0.95

        weather_multiplier = WEATHER_EFFECTS.get(weather_type, 1.0)
        base_range = available_energy / net_energy_per_km
        adjusted_range = base_range * weather_multiplier

        return max(0, adjusted_range)  # FIXED: No negative ranges

    df['solar_range_km'] = [
        calc_range(solar_tuktuk, terrain, speed, solar_kerb_weight, trickle_charge, solar_hours, weather_type)
        for terrain, speed in zip(terrains, speeds)
    ]

    df['grid_range_km'] = [
        calc_range(grid_tuktuk, terrain, speed, grid_kerb_weight, False, 0, weather_type)
        for terrain, speed in zip(terrains, speeds)
    ]

    # Weight breakdown
    base_weight = base_kerb_weight
    panel_weight = panel_area * panel_weight_per_m2
    capacitor_weight = capacitor_capacity * cap_weight_per_Wh
    total_solar_weight = base_weight + panel_weight + capacitor_weight

    solar_eff = df['solar_eff_Wh_per_km'].values
    grid_eff = df['grid_eff_Wh_per_km'].values

    # Efficiency and range plots (20x8 inches) - FIXED: No negative Y-axis
    fig_perf, (ax_eff, ax_range) = plt.subplots(1, 2, figsize=(20, 8))

    # Energy Efficiency (left) - Stacked bars
    bottom = np.zeros(len(df['terrain']))

    ax_eff.bar(df['terrain'], solar_eff * (base_weight / total_solar_weight), width=0.4, label=f'Solar (Base: {base_weight}kg)', align='center', color='#1f77b4', edgecolor='black', linewidth=1.5)
    bottom += solar_eff * (base_weight / total_solar_weight)
    ax_eff.bar(df['terrain'], solar_eff * (panel_weight / total_solar_weight), width=0.4, label=f'Solar (Panel: {panel_weight:.1f}kg)', align='center', bottom=bottom, color='#aec7e8', edgecolor='black', linewidth=1.5)
    bottom += solar_eff * (panel_weight / total_solar_weight)
    ax_eff.bar(df['terrain'], solar_eff * (capacitor_weight / total_solar_weight), width=0.4, label=f'Solar (Capacitor: {capacitor_weight:.1f}kg)', align='center', bottom=bottom, color='#4b78c9', edgecolor='black', linewidth=1.5)
    ax_eff.bar(df['terrain'], grid_eff, width=0.4, label=f'Grid TukTuk (Base: {base_weight}kg)', align='edge', color='#ff7f0e', edgecolor='black', linewidth=1.5)

    ax_eff.set_xlabel("Terrain Type", fontsize=14, fontweight='bold')
    ax_eff.set_ylabel("Energy Efficiency (Wh/km)", fontsize=14, fontweight='bold')
    ax_eff.set_title(f"Energy Efficiency Breakdown by Weight\nSolarTukTuk: {total_solar_weight:.1f}kg | Grid TukTuk: {base_weight}kg", fontsize=15, fontweight='bold')
    ax_eff.set_ylim(bottom=0)  # FIXED: Start Y-axis at 0
    ax_eff.legend(fontsize=12, loc='upper left')
    ax_eff.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax_eff.tick_params(axis='both', labelsize=12)

    # Range plot (right) - FIXED: No negative Y-axis
    bar_width = 0.35
    x = np.arange(len(df["terrain"]))

    solar_bars = ax_range.bar(x - bar_width / 2, df["solar_range_km"], width=bar_width, label="SolarTukTuk", color="#1f77b4", edgecolor='black', linewidth=1.5)
    grid_bars = ax_range.bar(x + bar_width / 2, df["grid_range_km"], width=bar_width, label="Grid TukTuk", color="#ff7f0e", edgecolor='black', linewidth=1.5)

    ax_range.set_xticks(x)
    ax_range.set_xticklabels(df["terrain"], fontsize=12)
    ax_range.set_xlabel("Terrain Type", fontsize=14, fontweight='bold')
    ax_range.set_ylabel("Estimated Range (km)", fontsize=14, fontweight='bold')
    ax_range.set_title(f"Estimated Range by Terrain\nWeather: {weather_type} | Trickle Charging: {'ON' if trickle_charge else 'OFF'}", fontsize=15, fontweight='bold')

    y_max = max(df["solar_range_km"].max(), df["grid_range_km"].max())
    ax_range.set_ylim(0, y_max * 1.15)  # FIXED: Explicitly start at 0
    ax_range.grid(True, linestyle="--", alpha=0.7, axis='y')
    ax_range.legend(fontsize=12)
    ax_range.tick_params(axis='both', labelsize=12)

    for bars in [solar_bars, grid_bars]:
        for bar in bars:
            height = bar.get_height()
            ax_range.text(bar.get_x() + bar.get_width() / 2, height + (y_max * 0.02), f"{height:.0f}", ha="center", va="bottom", fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Supercapacitor vs Battery lifecycle plot - FIXED: No negative Y-axis
    fig_cap_vs_bat, ax_cap2 = plt.subplots(figsize=(12, 7))
    cycle_counts = np.array([100, 500, 1000, 2000, 5000, 10000])
    supercap_efficiency = np.array([98, 97.5, 97, 96.5, 96, 95.5])
    battery_efficiency = np.array([95, 92, 88, 82, 75, 68])
    
    ax_cap2.plot(cycle_counts, supercap_efficiency, 'o-', linewidth=3, markersize=10, color='#A23B72', label='Supercapacitor')
    ax_cap2.plot(cycle_counts, battery_efficiency, 's-', linewidth=3, markersize=10, color='#2E86AB', label='Battery')
    ax_cap2.fill_between(cycle_counts, supercap_efficiency, battery_efficiency, alpha=0.25, color='#A23B72')
    ax_cap2.set_xlabel('Charge/Discharge Cycles', fontsize=12, fontweight='bold')
    ax_cap2.set_ylabel('Energy Efficiency (%)', fontsize=12, fontweight='bold')
    ax_cap2.set_title('Efficiency Over Lifecycle: Supercapacitor vs Battery', fontsize=14, fontweight='bold')
    ax_cap2.set_xscale('log')
    ax_cap2.set_ylim(bottom=0)  # FIXED: Start Y-axis at 0
    ax_cap2.grid(True, linestyle='--', alpha=0.6)
    ax_cap2.legend(fontsize=11, loc='lower left')
    ax_cap2.annotate('Supercapacitor maintains\n>95% efficiency', xy=(10000, 95.5), xytext=(3000, 80),
                     arrowprops=dict(arrowstyle='->', color='#A23B72', lw=2), fontsize=11, color='#A23B72', fontweight='bold')
    plt.tight_layout()

    # Battery degradation plot - FIXED: No negative Y-axis
    fig_degradation, ax_deg1 = plt.subplots(figsize=(12, 7))
    years_degradation = np.arange(0, years + 1)

    solar_degradation = 100 * np.exp(-0.03 * years_degradation)  # Slower degradation with supercaps
    grid_degradation = 100 * np.exp(-0.06 * years_degradation)

    ax_deg1.plot(years_degradation, solar_degradation, 'o-', linewidth=3, markersize=9, label='Solar TukTuk (Supercap-Assisted)', color='#1f77b4')
    ax_deg1.plot(years_degradation, grid_degradation, 's-', linewidth=3, markersize=9, label='Grid TukTuk (Battery Only)', color='#ff7f0e')

    ax_deg1.fill_between(years_degradation, solar_degradation, grid_degradation, alpha=0.25, color='#1f77b4', label='Capacity Advantage (Solar)')

    ax_deg1.axhline(80, color='red', linestyle='--', linewidth=2, label='80% Threshold (End of Life)')

    ax_deg1.set_title('Battery Capacity Degradation Over Time', fontsize=14, fontweight='bold')
    ax_deg1.set_xlabel('Years of Operation', fontsize=12, fontweight='bold')
    ax_deg1.set_ylabel('Remaining Capacity (%)', fontsize=12, fontweight='bold')
    ax_deg1.set_ylim(bottom=0)  # FIXED: Start Y-axis at 0
    ax_deg1.grid(True, linestyle='--', alpha=0.6)
    ax_deg1.legend(fontsize=10, loc='lower left')

    solar_80_year = -np.log(0.8) / 0.03
    grid_80_year = -np.log(0.8) / 0.06

    ax_deg1.annotate(f'Solar hits 80%\nat ~{solar_80_year:.1f} years', xy=(solar_80_year, 80), xytext=(solar_80_year + 1.5, 65),
                     arrowprops=dict(arrowstyle='->', lw=2, color='#1f77b4'), fontsize=11, fontweight='bold', color='#1f77b4')
    ax_deg1.annotate(f'Grid hits 80%\nat ~{grid_80_year:.1f} years', xy=(grid_80_year, 80), xytext=(grid_80_year - 1, 92),
                     arrowprops=dict(arrowstyle='->', lw=2, color='#ff7f0e'), fontsize=11, fontweight='bold', color='#ff7f0e')

    plt.tight_layout()

    # Cost DataFrame
    df_cost = pd.DataFrame({
        "Terrain": df['terrain'],
        "Solar Cost/km (USD)": df['solar_cost_per_km'].round(4),
        "Grid Cost/km (USD)": df['grid_cost_per_km'].round(4),
        "Solar Total 1st Year (USD)": (df['solar_cost_per_km'] * annual_km + initial_investment_solar).round(2),
        "Grid Total 1st Year (USD)": (df['grid_cost_per_km'] * annual_km + initial_investment_electric).round(2)
    })

    return {
        'weather_impact_plot': fig_weather,
        'carbon_plot': fig_carbon,
        'efficiency_plot': fig_perf,
        'cost_plot': fig_cost,
        'supercap_vs_battery_plot': fig_cap_vs_bat,
        'degradation_plot': fig_degradation,
        'df_efficiency': df[['terrain', 'solar_eff_Wh_per_km', 'grid_eff_Wh_per_km']].rename(columns={
            "terrain": "Terrain",
            "solar_eff_Wh_per_km": "SolarTukTuk (Wh/km)",
            "grid_eff_Wh_per_km": "Grid TukTuk (Wh/km)"
        }).round(2),
        'df_range': df[['terrain', 'solar_range_km', 'grid_range_km']].rename(columns={
            "terrain": "Terrain",
            "solar_range_km": "SolarTukTuk Range (km)",
            "grid_range_km": "Grid TukTuk Range (km)"
        }).round(1),
        'df_cost': df_cost,
        'total_solar_power_generated_Wh': total_solar_power_generated,
        'carbon_emissions_kg_grid': carbon_emissions_grid,
        'carbon_emissions_kg_solar': carbon_emissions_solar
    }

if __name__ == "__main__":
    results = run_simulation(weather_type="Clear")
    results['weather_impact_plot'].show()
    results['carbon_plot'].show()
    results['efficiency_plot'].show()
    results['cost_plot'].show()
    results['supercap_vs_battery_plot'].show()
    results['degradation_plot'].show()
    plt.show()
