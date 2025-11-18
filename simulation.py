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
    def __init__(self, efficiency, area):
        self.base_efficiency = efficiency
        self.area = area

    def generate_power(self, sunlight_intensity):
        efficiency_factor = 1 - min(0.15, max(0, (sunlight_intensity - 800) / 5000))
        actual_efficiency = self.base_efficiency * efficiency_factor
        return actual_efficiency * self.area * sunlight_intensity

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
        self.terrain_energy_usage = {
            "Flat": 0,
            "Hill": 0,
            "Sandy": 0,
            "Rough": 0,
            "Downhill": 0
        }

    def drive(self, distance, terrain, speed=None):
        terrain_factors = {
            "Flat": 1.0,
            "Hill": 1.5,
            "Sandy": 1.8,
            "Rough": 2.0,
            "Downhill": 0.7
        }
        terrain_factor = terrain_factors.get(terrain, 1.0)
        energy_needed = self.motor.energy_required(distance, terrain_factor, self.kerb_weight)

        if self.battery.charge_level >= energy_needed:
            self.battery.discharge(energy_needed)
            self.total_energy_consumed += energy_needed
            self.total_distance_covered += distance
            self.terrain_energy_usage[terrain] += energy_needed
            wh_per_km = self.total_energy_consumed / self.total_distance_covered
            print(f"Energy Efficiency: {wh_per_km:.2f} Wh/km | Terrain: {terrain}")
        else:
            print("Warning: Not enough battery to complete trip!")

        if terrain == "Downhill" and speed is not None:
            self.regenerative_braking(speed)

    def regenerative_braking(self, speed):
        recovered_energy = self.motor.power_rating * speed * 0.05
        self.battery.charge(recovered_energy)
        print(f"Regenerative braking recovered {recovered_energy:.2f} Wh")

class SolarTukTuk:
    def __init__(
        self,
        battery_capacity,
        capacitor_capacity,
        motor,
        kerb_weight,
        top_speed,
        panel_area,
        panel_efficiency
    ):
        self.battery = Battery(battery_capacity)
        self.supercapacitor = Supercapacitor(capacitor_capacity)
        self.motor = motor
        self.kerb_weight = kerb_weight
        self.top_speed = top_speed
        self.solar_panel = SolarPanel(panel_efficiency, panel_area)
        self.weather = Weather()
        self.total_energy_consumed = 0
        self.total_distance_covered = 0
        self.terrain_energy_usage = {
            "Flat": 0,
            "Hill": 0,
            "Sandy": 0,
            "Rough": 0,
            "Downhill": 0
        }
        self.hourly_data = []

    def drive(self, distance, terrain, speed=None):
        terrain_factors = {
            "Flat": 1.0,
            "Hill": 1.5,
            "Sandy": 1.8,
            "Rough": 2.0,
            "Downhill": 0.7
        }
        terrain_factor = terrain_factors.get(terrain, 1.0)
        energy_needed = self.motor.energy_required(distance, terrain_factor, self.kerb_weight)

        boost_needed = min(energy_needed * 0.2, self.supercapacitor.charge_level)
        actual_boost = self.supercapacitor.provide_boost(boost_needed)
        remaining_energy_needed = energy_needed - actual_boost

        total_available = self.battery.charge_level + self.supercapacitor.charge_level

        if total_available >= remaining_energy_needed:
            battery_ratio = self.battery.charge_level / total_available if total_available > 0 else 0
            battery_usage = remaining_energy_needed * battery_ratio

            self.battery.discharge(battery_usage)
            self.total_energy_consumed += energy_needed
            self.total_distance_covered += distance
            self.terrain_energy_usage[terrain] += energy_needed

            wh_per_km = self.total_energy_consumed / self.total_distance_covered
            print(f"Energy Efficiency: {wh_per_km:.2f} Wh/km | Terrain: {terrain}")
        else:
            print("Warning: Not enough energy to complete trip!")

        if terrain == "Downhill" and speed is not None:
            self.regenerative_braking(speed)

    def regenerative_braking(self, speed):
        recovered_energy = self.motor.power_rating * speed * 0.05
        self.battery.charge(recovered_energy)
        print(f"Regenerative braking recovered {recovered_energy:.2f} Wh")

    def charge_solar(self, duration_hours):
        total_charge = 0
        for hour in range(duration_hours):
            sunlight_intensity = self.weather.get_sunlight(hour + 6)
            for minute in range(60):
                solar_input = self.solar_panel.generate_power(sunlight_intensity) / 60
                self.battery.charge(solar_input)
                total_charge += solar_input
            self.hourly_data.append((hour + 6, self.battery.charge_level))
        print(f"Battery trickle charged by {total_charge:.2f} Wh over {duration_hours} hours")

WEATHER_EFFECTS = {
    "Clear": 1.0,
    "Cloudy": 0.95,
    "Rainy": 0.85,
    "Windy": 0.9
}

SOLAR_EMISSION_FACTOR = 0.05
GRID_EMISSION_FACTOR = 0.4

def run_simulation(
    battery_capacity=5000,
    capacitor_capacity=500,
    motor_power=5000,
    motor_efficiency=0.85,
    panel_area=1.5,
    panel_efficiency=0.2,
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

    solar_kerb_weight = (
        base_kerb_weight +
        (panel_area * panel_weight_per_m2) +
        (capacitor_capacity * cap_weight_per_Wh)
    )
    grid_kerb_weight = base_kerb_weight

    motor = Motor(power_rating=motor_power, efficiency=motor_efficiency)

    solar_tuktuk = SolarTukTuk(
        battery_capacity=battery_capacity,
        capacitor_capacity=capacitor_capacity,
        motor=motor,
        kerb_weight=solar_kerb_weight,
        top_speed=80,
        panel_area=panel_area,
        panel_efficiency=panel_efficiency
    )

    total_charge = 0
    hourly_irradiance = []
    hourly_power_output = []
    
    # Display full day from 6am to 6pm (18:00)
    full_day_hours = range(6, 19)  # 6am to 6pm
    
    for hour in full_day_hours:
        sunlight_intensity = solar_tuktuk.weather.get_sunlight(hour)
        hourly_irradiance.append(sunlight_intensity)
        
        # Calculate actual power output from solar panel
        power_output = solar_tuktuk.solar_panel.generate_power(sunlight_intensity)
        hourly_power_output.append(power_output)
        
        # Only charge during solar_hours for actual simulation
        if hour < 6 + solar_hours:
            for minute in range(60):
                solar_input = power_output / 60
                solar_tuktuk.battery.charge(solar_input)
                total_charge += solar_input

    weather_multiplier = WEATHER_EFFECTS.get(weather_type, 1.0)
    total_solar_power_generated = total_charge * weather_multiplier

    # Enhanced weather impact plot
    fig_weather, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    hours_arr = np.array(list(full_day_hours))
    
    # Plot 1: Sunlight Intensity
    ax1.plot(hours_arr, hourly_irradiance, 'o-', linewidth=2, markersize=8, 
             color='#FDB462', label="Sunlight Intensity")
    ax1.fill_between(hours_arr, hourly_irradiance, alpha=0.3, color='#FDB462')
    ax1.set_xlabel("Hour of Day", fontsize=11)
    ax1.set_ylabel("Sunlight Intensity (W/m²)", fontsize=11)
    ax1.set_title(f"Sunlight Intensity Throughout the Day\nWeather: {weather_type}", 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xticks(hours_arr)
    ax1.set_xticklabels([f"{h}:00" for h in hours_arr], rotation=45)
    ax1.legend(fontsize=10)
    
    # Highlight rainy hours if any
    rainy_hours = solar_tuktuk.weather.rainy_hours
    for rh in rainy_hours:
        if rh in full_day_hours:
            ax1.axvspan(rh - 0.3, rh + 0.3, alpha=0.2, color='blue', label='Rainy' if rh == min(rainy_hours) else '')
    
    # Plot 2: Power Output
    ax2.plot(hours_arr, hourly_power_output, 's-', linewidth=2, markersize=8,
             color='#1f77b4', label="Solar Power Output")
    ax2.fill_between(hours_arr, hourly_power_output, alpha=0.3, color='#1f77b4')
    ax2.set_xlabel("Hour of Day", fontsize=11)
    ax2.set_ylabel("Power Output (W)", fontsize=11)
    ax2.set_title(f"Solar Panel Power Output (Panel Area: {panel_area}m², Efficiency: {panel_efficiency*100}%)",
                  fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_xticks(hours_arr)
    ax2.set_xticklabels([f"{h}:00" for h in hours_arr], rotation=45)
    ax2.legend(fontsize=10)
    
    # Add weather multiplier info
    fig_weather.text(0.99, 0.01, f"Weather Multiplier: {weather_multiplier:.2f}x | "
                                  f"Total Energy Generated: {total_solar_power_generated:.2f} Wh",
                     ha='right', va='bottom', fontsize=10, style='italic')
    
    plt.tight_layout()

    grid_tuktuk = GridTukTuk(
        battery_capacity=battery_capacity,
        motor=motor,
        kerb_weight=grid_kerb_weight,
        top_speed=80
    )

    results = []

    for terrain, speed in zip(terrains, speeds):
        distance = distance_per_terrain

        before_energy_solar = solar_tuktuk.battery.charge_level + solar_tuktuk.supercapacitor.charge_level
        solar_tuktuk.drive(
            distance=distance,
            terrain=terrain,
            speed=speed if terrain == "Downhill" else None
        )
        after_energy_solar = solar_tuktuk.battery.charge_level + solar_tuktuk.supercapacitor.charge_level
        net_energy_used_solar = before_energy_solar - after_energy_solar

        before_energy_grid = grid_tuktuk.battery.charge_level
        grid_tuktuk.drive(
            distance=distance,
            terrain=terrain,
            speed=speed if terrain == "Downhill" else None
        )
        after_energy_grid = grid_tuktuk.battery.charge_level
        net_energy_used_grid = before_energy_grid - after_energy_grid

        results.append({
            "terrain": terrain,
            "distance_km": distance,
            "solar_eff_Wh_per_km": net_energy_used_solar / distance if distance > 0 else None,
            "grid_eff_Wh_per_km": net_energy_used_grid / distance if distance > 0 else None,
        })

    df = pd.DataFrame(results)

    # Carbon emissions calculations
    solar_energy_kwh = total_solar_power_generated / 1000
    grid_energy_kwh = (df['grid_eff_Wh_per_km'].mean() * annual_km) / 1000
    carbon_emissions_solar = solar_energy_kwh * SOLAR_EMISSION_FACTOR
    carbon_emissions_grid = grid_energy_kwh * GRID_EMISSION_FACTOR

    fig_carbon, ax_carbon = plt.subplots(figsize=(12, 6))
    labels = ["Solar TukTuk", "Grid TukTuk"]
    values = [carbon_emissions_solar, carbon_emissions_grid]
    ax_carbon.bar(labels, values, color=['#1f77b4', '#ff7f0e'])
    ax_carbon.set_ylabel("kg CO₂ Emitted per Year")
    ax_carbon.set_title("Carbon Emissions Comparison (kg CO₂)")
    ax_carbon.grid(True, linestyle='--', alpha=0.6)

    if trickle_charge:
        solar_tuktuk.charge_solar(duration_hours=solar_hours)

    # --- COST ANALYSIS ---
    df['solar_cost_per_km'] = df['solar_eff_Wh_per_km'] / 1000 * 0   # Free sunlight
    df['grid_cost_per_km'] = df['grid_eff_Wh_per_km'] / 1000 * grid_cost_per_kwh

    avg_grid_cost_per_km = df['grid_cost_per_km'].mean()

    years_arr = np.arange(0, years + 1)

    # Solar = one-time investment only
    solar_total_cost = np.full_like(years_arr, initial_investment_solar, dtype=float)

    # Grid = initial + yearly cost growth
    grid_total_cost = initial_investment_electric + (avg_grid_cost_per_km * annual_km * years_arr)

    # --- TOTAL COST PLOT ---
    fig_cost, ax_cost = plt.subplots(figsize=(8, 4))
    ax_cost.plot(
        years_arr,
        solar_total_cost,
        label='Solar TukTuk (No Grid Cost)',
        linewidth=2.5,
        color='#1f77b4'
    )
    ax_cost.plot(
        years_arr,
        grid_total_cost,
        label='Grid TukTuk (Energy Cost Accumulation)',
        linewidth=2.5,
        color='#ff7f0e'
    )

    ax_cost.fill_between(years_arr, solar_total_cost, grid_total_cost, color='#ff7f0e', alpha=0.1, label="Cost Savings with Solar")

    ax_cost.set_xlabel('Years of Operation')
    ax_cost.set_ylabel('Total Cost (USD)')
    ax_cost.set_title(f'Total Cost Projection over {years} Years')
    ax_cost.legend()
    ax_cost.grid(True, linestyle='--', alpha=0.7)

    ax_cost.text(years_arr[-1], solar_total_cost[-1], f"${solar_total_cost[-1]:,.0f}", va='bottom', ha='right', color='#1f77b4')
    ax_cost.text(years_arr[-1], grid_total_cost[-1], f"${grid_total_cost[-1]:,.0f}", va='top', ha='right', color='#ff7f0e')

    def calc_range(vehicle, terrain, speed, kerb_weight, trickle_charge, solar_hours=6):
        if trickle_charge and hasattr(vehicle, 'charge_solar'):
            vehicle.charge_solar(duration_hours=solar_hours)

        terrain_factors = {
            "Flat": 1.0,
            "Hill": 1.5,
            "Sandy": 1.8,
            "Rough": 2.0,
            "Downhill": 0.7
        }
        terrain_factor = terrain_factors[terrain]

        energy_per_km = vehicle.motor.energy_required(1, terrain_factor, kerb_weight=kerb_weight)

        if terrain == "Downhill":
            recovered_per_km = min(
                energy_per_km * 0.1,
                vehicle.motor.power_rating * speed * 0.005
            )
        else:
            recovered_per_km = 0

        net_energy_per_km = max(energy_per_km - recovered_per_km, 0.01)

        if hasattr(vehicle, 'supercapacitor'):
            available_energy = vehicle.battery.charge_level + vehicle.supercapacitor.charge_level
        else:
            available_energy = vehicle.battery.charge_level

        weather_multiplier = WEATHER_EFFECTS.get(weather_type, 1.0)
        base_range = available_energy / net_energy_per_km
        adjusted_range = base_range * weather_multiplier

        return adjusted_range

    df['solar_range_km'] = [
        calc_range(
            SolarTukTuk(
                battery_capacity=battery_capacity,
                capacitor_capacity=capacitor_capacity,
                motor=motor,
                kerb_weight=solar_kerb_weight,
                top_speed=80,
                panel_area=panel_area,
                panel_efficiency=panel_efficiency
            ),
            terrain,
            speed,
            solar_kerb_weight,
            trickle_charge,
            solar_hours
        )
        for terrain, speed in zip(terrains, speeds)
    ]

    df['grid_range_km'] = [
        calc_range(
            GridTukTuk(
                battery_capacity=battery_capacity,
                motor=motor,
                kerb_weight=grid_kerb_weight,
                top_speed=80
            ),
            terrain,
            speed,
            grid_kerb_weight,
            False,
            0
        )
        for terrain, speed in zip(terrains, speeds)
    ]

    base_weight = base_kerb_weight
    panel_weight = panel_area * panel_weight_per_m2
    capacitor_weight = capacitor_capacity * cap_weight_per_Wh

    solar_eff = df['solar_eff_Wh_per_km'].values
    grid_eff = df['grid_eff_Wh_per_km'].values

    total_solar_weight = base_weight + panel_weight + capacitor_weight
    solar_eff_base = solar_eff * (base_weight / total_solar_weight)
    solar_eff_panel = solar_eff * (panel_weight / total_solar_weight)
    solar_eff_cap = solar_eff * (capacitor_weight / total_solar_weight)

    fig_eff, ax_eff = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(df['terrain']))

    ax_eff.bar(
        df['terrain'],
        solar_eff_base,
        width=0.4,
        label=f'Solar (Base: {base_weight}kg)',
        align='center',
        color='#1f77b4'
    )

    bottom += solar_eff_base
    ax_eff.bar(
        df['terrain'],
        solar_eff_panel,
        width=0.4,
        label=f'Solar (Panel: {panel_weight:.1f}kg)',
        align='center',
        bottom=bottom,
        color='#aec7e8'
    )

    bottom += solar_eff_panel
    ax_eff.bar(
        df['terrain'],
        solar_eff_cap,
        width=0.4,
        label=f'Solar (Capacitor: {capacitor_weight:.1f}kg)',
        align='center',
        bottom=bottom,
        color='#4b78c9'
    )

    ax_eff.bar(
        df['terrain'],
        grid_eff,
        width=0.4,
        label=f'Grid TukTuk (Base: {base_weight}kg)',
        align='edge',
        color='#ff7f0e'
    )

    ax_eff.set_xlabel("Terrain Type")
    ax_eff.set_ylabel("Energy Efficiency (Wh/km)")
    ax_eff.set_title(
        f"Energy Efficiency Breakdown by Weight Components\n"
        f"SolarTukTuk Total Weight: {total_solar_weight:.1f}kg vs "
        f"Grid TukTuk: {base_weight}kg"
    )
    ax_eff.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_eff.grid(True, linestyle='--', alpha=0.7)
    ax_eff.set_ylim(0, max(solar_eff.max(), grid_eff.max()) * 1.2)
    plt.tight_layout()

    fig_range, ax_range = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(df["terrain"]))

    solar_bars = ax_range.bar(
        x - bar_width / 2,
        df["solar_range_km"],
        width=bar_width,
        label="SolarTukTuk",
        color="#1f77b4"
    )

    grid_bars = ax_range.bar(
        x + bar_width / 2,
        df["grid_range_km"],
        width=bar_width,
        label="Grid TukTuk",
        color="#ff7f0e"
    )

    ax_range.set_xticks(x)
    ax_range.set_xticklabels(df["terrain"], fontsize=10)
    ax_range.set_xlabel("Terrain Type")
    ax_range.set_ylabel("Estimated Range (km)")
    ax_range.set_title(
        f"Vehicle Range by Terrain and Weather Type: {weather_type}\n"
        f"(Trickle Charging: {'ON' if trickle_charge else 'OFF'})"
    )

    y_max = max(df["solar_range_km"].max(), df["grid_range_km"].max())
    ax_range.set_ylim(0, y_max * 1.15)
    ax_range.grid(True, linestyle="--", alpha=0.7)
    ax_range.legend()

    for bars in [solar_bars, grid_bars]:
        for bar in bars:
            height = bar.get_height()
            ax_range.text(
                bar.get_x() + bar.get_width() / 2,
                height + (y_max * 0.02),
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()

    # --- SUPERCAPACITOR VS BATTERY EFFICIENCY PLOT ---
    fig_cap_vs_bat, (ax_cap1, ax_cap2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Energy source breakdown for each terrain
    terrains_arr = df['terrain'].values
    x_pos = np.arange(len(terrains_arr))
    
    # Calculate energy distribution (simplified model)
    # Supercapacitor provides burst power (20% of energy needs)
    # Battery provides sustained power (80% of energy needs)
    solar_total_energy = df['solar_eff_Wh_per_km'].values * distance_per_terrain
    cap_contribution = solar_total_energy * 0.2  # 20% from supercapacitor
    battery_contribution = solar_total_energy * 0.8  # 80% from battery
    
    width = 0.35
    ax_cap1.bar(x_pos - width/2, battery_contribution, width, 
                label='Battery', color='#2E86AB', alpha=0.8)
    ax_cap1.bar(x_pos - width/2, cap_contribution, width, 
                bottom=battery_contribution, label='Supercapacitor Boost', 
                color='#A23B72', alpha=0.8)
    
    # Grid tuk-tuk (battery only)
    grid_total_energy = df['grid_eff_Wh_per_km'].values * distance_per_terrain
    ax_cap1.bar(x_pos + width/2, grid_total_energy, width, 
                label='Grid Battery Only', color='#F18F01', alpha=0.8)
    
    ax_cap1.set_xlabel('Terrain Type', fontsize=11)
    ax_cap1.set_ylabel('Energy per Trip (Wh)', fontsize=11)
    ax_cap1.set_title('Energy Source Distribution\nSolar (Battery+Supercap) vs Grid (Battery Only)', 
                      fontsize=12, fontweight='bold')
    ax_cap1.set_xticks(x_pos)
    ax_cap1.set_xticklabels(terrains_arr, rotation=45)
    ax_cap1.legend(fontsize=9)
    ax_cap1.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # Right plot: Efficiency comparison (charge/discharge cycles)
    cycle_counts = np.array([100, 500, 1000, 2000, 5000, 10000])
    
    # Supercapacitor: Very high cycle life, minimal degradation
    supercap_efficiency = np.array([98, 97.5, 97, 96.5, 96, 95.5])
    
    # Battery: Degrades more with cycles
    battery_efficiency = np.array([95, 92, 88, 82, 75, 68])
    
    ax_cap2.plot(cycle_counts, supercap_efficiency, 'o-', linewidth=2.5, 
                 markersize=8, color='#A23B72', label='Supercapacitor')
    ax_cap2.plot(cycle_counts, battery_efficiency, 's-', linewidth=2.5, 
                 markersize=8, color='#2E86AB', label='Battery')
    
    ax_cap2.fill_between(cycle_counts, supercap_efficiency, battery_efficiency, 
                         alpha=0.2, color='#A23B72')
    
    ax_cap2.set_xlabel('Charge/Discharge Cycles', fontsize=11)
    ax_cap2.set_ylabel('Energy Efficiency (%)', fontsize=11)
    ax_cap2.set_title('Efficiency Over Lifecycle\nSupercapacitor vs Battery', 
                      fontsize=12, fontweight='bold')
    ax_cap2.set_xscale('log')
    ax_cap2.grid(True, linestyle='--', alpha=0.6)
    ax_cap2.legend(fontsize=10)
    ax_cap2.set_ylim(60, 100)
    
    # Add annotation
    ax_cap2.annotate('Supercapacitor maintains\n>95% efficiency', 
                     xy=(10000, 95.5), xytext=(5000, 85),
                     arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5),
                     fontsize=9, color='#A23B72', fontweight='bold')
    
    plt.tight_layout()
    
    # --- BATTERY DEGRADATION COMPARISON PLOT ---
    fig_degradation, (ax_deg1, ax_deg2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Capacity degradation over years
    years_degradation = np.arange(0, years + 1)
    
    # Solar TukTuk with supercapacitor: Reduced battery stress
    # Supercapacitor handles high-power demands, reducing battery cycling stress
    solar_degradation = 100 * np.exp(-0.03 * years_degradation)  # Slower degradation
    
    # Grid TukTuk: Higher battery stress (all loads on battery)
    grid_degradation = 100 * np.exp(-0.06 * years_degradation)  # Faster degradation
    
    ax_deg1.plot(years_degradation, solar_degradation, 'o-', linewidth=2.5,
                 markersize=8, color='#1f77b4', label='Solar TukTuk (with Supercap)')
    ax_deg1.plot(years_degradation, grid_degradation, 's-', linewidth=2.5,
                 markersize=8, color='#ff7f0e', label='Grid TukTuk (Battery Only)')
    
    ax_deg1.fill_between(years_degradation, solar_degradation, grid_degradation,
                         alpha=0.2, color='#1f77b4', label='Capacity Advantage')
    
    ax_deg1.axhline(y=80, color='red', linestyle='--', linewidth=1.5, 
                    label='80% Threshold (End of Life)')
    
    ax_deg1.set_xlabel('Years of Operation', fontsize=11)
    ax_deg1.set_ylabel('Battery Capacity (%)', fontsize=11)
    ax_deg1.set_title('Battery Capacity Degradation Over Time\nSolar (Supercap-assisted) vs Grid', 
                      fontsize=12, fontweight='bold')
    ax_deg1.grid(True, linestyle='--', alpha=0.6)
    ax_deg1.legend(fontsize=9, loc='lower left')
    ax_deg1.set_ylim(50, 105)
    
    # Find when each reaches 80% capacity
    solar_80_year = -np.log(0.8) / 0.03
    grid_80_year = -np.log(0.8) / 0.06
    
    ax_deg1.annotate(f'Solar: ~{solar_80_year:.1f} years to 80%',
                     xy=(solar_80_year, 80), xytext=(solar_80_year + 1, 70),
                     arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
                     fontsize=9, color='#1f77b4', fontweight='bold')
    
    ax_deg1.annotate(f'Grid: ~{grid_80_year:.1f} years to 80%',
                     xy=(grid_80_year, 80), xytext=(grid_80_year - 2, 90),
                     arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5),
                     fontsize=9, color='#ff7f0e', fontweight='bold')
    
    # Right plot: Cumulative cost including battery replacement
    battery_replacement_cost = 2000  # USD per replacement
    
    solar_replacement_years = []
    grid_replacement_years = []
    
    # Calculate replacement years (when capacity drops below 80%)
    for year in years_degradation:
        if 100 * np.exp(-0.03 * year) < 80 and year > 0:
            solar_replacement_years.append(year)
        if 100 * np.exp(-0.06 * year) < 80 and year > 0:
            grid_replacement_years.append(year)
    
    # Only keep first replacement for each
    solar_replacement_year = min(solar_replacement_years) if solar_replacement_years else None
    grid_replacement_year = min(grid_replacement_years) if grid_replacement_years else None
    
    # Calculate total cost including replacements
    solar_total_with_replacement = solar_total_cost.copy()
    grid_total_with_replacement = grid_total_cost.copy()
    
    if solar_replacement_year and solar_replacement_year <= years:
        solar_total_with_replacement[int(solar_replacement_year):] += battery_replacement_cost
    
    if grid_replacement_year and grid_replacement_year <= years:
        grid_total_with_replacement[int(grid_replacement_year):] += battery_replacement_cost
    
    ax_deg2.plot(years_arr, solar_total_with_replacement, 'o-', linewidth=2.5,
                 markersize=6, color='#1f77b4', label='Solar TukTuk')
    ax_deg2.plot(years_arr, grid_total_with_replacement, 's-', linewidth=2.5,
                 markersize=6, color='#ff7f0e', label='Grid TukTuk')
    
    # Mark replacement points
    if solar_replacement_year and solar_replacement_year <= years:
        idx = int(solar_replacement_year)
        ax_deg2.plot(idx, solar_total_with_replacement[idx], 'X', 
                    markersize=15, color='red', 
                    label=f'Battery Replacement (Year {idx})')
    
    if grid_replacement_year and grid_replacement_year <= years:
        idx = int(grid_replacement_year)
        ax_deg2.plot(idx, grid_total_with_replacement[idx], 'X', 
                    markersize=15, color='darkred')
    
    ax_deg2.set_xlabel('Years of Operation', fontsize=11)
    ax_deg2.set_ylabel('Total Cost Including Replacements (USD)', fontsize=11)
    ax_deg2.set_title(f'Total Cost with Battery Replacement\n(Replacement Cost: ${battery_replacement_cost})', 
                      fontsize=12, fontweight='bold')
    ax_deg2.grid(True, linestyle='--', alpha=0.6)
    ax_deg2.legend(fontsize=9)
    
    # Add final cost labels
    ax_deg2.text(years_arr[-1], solar_total_with_replacement[-1], 
                 f"${solar_total_with_replacement[-1]:,.0f}", 
                 va='bottom', ha='right', color='#1f77b4', fontweight='bold')
    ax_deg2.text(years_arr[-1], grid_total_with_replacement[-1], 
                 f"${grid_total_with_replacement[-1]:,.0f}", 
                 va='top', ha='right', color='#ff7f0e', fontweight='bold')
    
    plt.tight_layout()

    df_cost = pd.DataFrame({
        "Terrain": df['terrain'],
        "Solar Cost/km (USD)": df['solar_cost_per_km'].round(4),
        "Grid Cost/km (USD)": df['grid_cost_per_km'].round(4),
        "Solar Total 1st Year (USD)": (
            df['solar_cost_per_km'] * annual_km + initial_investment_solar
        ).round(2),
        "Grid Total 1st Year (USD)": (
            df['grid_cost_per_km'] * annual_km + initial_investment_electric
        ).round(2)
    })

    return {
        'weather_impact_plot': fig_weather,
        'carbon_plot': fig_carbon,
        'efficiency_plot': fig_eff,
        'range_plot': fig_range,
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
    results['range_plot'].show()
    results['cost_plot'].show()
    results['supercap_vs_battery_plot'].show()
    results['degradation_plot'].show()
    plt.show()