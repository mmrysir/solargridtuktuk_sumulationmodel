#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simpy import Environment, Resource
import simpy
from sklearn.metrics import mean_squared_error


# In[3]:


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


# In[4]:


class Battery:
    def __init__(self, capacity):
        self.capacity = capacity
        self.charge_level = capacity * 0.5

    def charge(self, power_input):
        self.charge_level = min(self.capacity, self.charge_level + power_input)

    def discharge(self, power_output):
        self.charge_level = max(0, self.charge_level - power_output)


# In[5]:


class SolarPanel:
    def __init__(self, efficiency, area):
        self.base_efficiency = efficiency
        self.area = area

    def generate_power(self, sunlight_intensity):
        efficiency_factor = 1 - min(0.15, (sunlight_intensity - 800) / 5000)
        actual_efficiency = self.base_efficiency * efficiency_factor
        return actual_efficiency * self.area * sunlight_intensity


# In[6]:


class Supercapacitor:
    def __init__(self, capacity):
        self.capacity = capacity
        self.charge_level = capacity * 0.5

    def provide_boost(self, boost_power):
        boost_available = min(self.charge_level, boost_power)
        self.charge_level -= boost_available
        return boost_available


# In[7]:


class Motor:
    def __init__(self, power_rating, efficiency):
        self.power_rating = power_rating  # in Watts
        self.efficiency = efficiency      # as a fraction (e.g., 0.85)

    def energy_required(self, distance, terrain_factor, kerb_weight=400):
        """
        Calculate energy required (Wh) for a trip, scaled by kerb weight.
        - distance: trip distance in km
        - terrain_factor: multiplier for terrain difficulty
        - kerb_weight: vehicle weight in kg (default 400kg)
        """
        # Scale energy linearly based on kerb weight (using 400kg as the reference)
        weight_factor = kerb_weight / 400
        base_energy = (self.power_rating * distance * terrain_factor) / 1000 / self.efficiency
        return base_energy * weight_factor


# In[8]:


import random


# In[9]:


class ElectricTukTuk:
    def __init__(self, battery_capacity, motor, kerb_weight, top_speed):
        self.battery = Battery(battery_capacity)
        self.motor = motor
        self.kerb_weight = kerb_weight
        self.top_speed = top_speed
        self.total_energy_consumed = 0
        self.total_distance_covered = 0
        self.terrain_energy_usage = {"Flat": 0, "Hill": 0, "Sandy": 0, "Rough": 0, "Downhill": 0}
        self.hourly_data = []

    def drive(self, distance, terrain, speed=None):
        terrain_factor = {"Flat": 1.0, "Hill": 1.5, "Sandy": 1.8, "Rough": 2.0, "Downhill": 0.7}.get(terrain, 1.0)
        energy_needed = self.motor.energy_required(distance, terrain_factor)
        if self.battery.charge_level >= energy_needed:
            self.battery.discharge(energy_needed)
            self.total_energy_consumed += energy_needed
            self.total_distance_covered += distance
            self.terrain_energy_usage[terrain] += energy_needed
            wh_per_km = self.total_energy_consumed / self.total_distance_covered
            print(f"Energy Efficiency: {wh_per_km:.2f} Wh/km | Terrain: {terrain}")
        else:
            print("Warning: Not enough battery to complete trip!")

        # Optionally trigger regenerative braking if going downhill and speed is given
        if terrain == "Downhill" and speed is not None:
            self.regenerative_braking(speed, terrain)

    def regenerative_braking(self, speed, terrain):
        if terrain == "Downhill":
            motor_power = getattr(self.motor, 'power_rating', 5000)
            recovered_energy = (motor_power * speed * 0.05)
            self.battery.charge(recovered_energy)
            print(f"Regenerative braking recovered {recovered_energy:.2f} Wh")


# In[10]:


class SolarTukTuk:
    def __init__(self, battery_capacity, capacitor_capacity, motor, kerb_weight, top_speed, panel_area, panel_efficiency):
        self.battery = Battery(battery_capacity)
        self.supercapacitor = Supercapacitor(capacitor_capacity)
        self.motor = motor  # Instance of Motor class
        self.kerb_weight = kerb_weight
        self.top_speed = top_speed
        self.solar_panel = SolarPanel(panel_efficiency, panel_area)
        self.weather = Weather()
        self.total_energy_consumed = 0
        self.total_distance_covered = 0
        self.terrain_energy_usage = {"Flat": 0, "Hill": 0, "Sandy": 0, "Rough": 0, "Downhill": 0}
        self.hourly_data = []

    def drive(self, distance, terrain, speed=None):
        terrain_factor = {"Flat": 1.0, "Hill": 1.5, "Sandy": 1.8, "Rough": 2.0, "Downhill": 0.7}.get(terrain, 1.0)
        energy_needed = self.motor.energy_required(distance, terrain_factor)
        boost_needed = min(energy_needed * 0.2, self.supercapacitor.charge_level)

        self.supercapacitor.provide_boost(boost_needed)
        remaining_energy_needed = energy_needed - boost_needed
        total_available = self.battery.charge_level + self.supercapacitor.charge_level

        if total_available >= remaining_energy_needed:
            battery_ratio = self.battery.charge_level / total_available
            battery_usage = remaining_energy_needed * battery_ratio

            self.battery.discharge(battery_usage)
            self.total_energy_consumed += remaining_energy_needed
            self.total_distance_covered += distance
            self.terrain_energy_usage[terrain] += remaining_energy_needed

            wh_per_km = self.total_energy_consumed / self.total_distance_covered
            print(f"Energy Efficiency: {wh_per_km:.2f} Wh/km | Terrain: {terrain}")
        else:
            print("Warning: Not enough energy to complete trip!")

        # Optionally trigger regenerative braking if going downhill and speed is given
        if terrain == "Downhill" and speed is not None:
            self.regenerative_braking(speed, terrain)

    def regenerative_braking(self, speed, terrain):
        if terrain == "Downhill":
            # Use motor's power_rating if available, otherwise use a default value
            motor_power = getattr(self.motor, 'power_rating', 5000)
            recovered_energy = (motor_power * speed * 0.05)
            self.battery.charge(recovered_energy)
            print(f"Regenerative braking recovered {recovered_energy:.2f} Wh")

    def charge_solar(self, duration_hours):
        total_charge = 0
        for hour in range(duration_hours):
            sunlight_intensity = self.weather.get_sunlight(hour + 6)
            for minute in range(60):
                solar_input = (self.solar_panel.generate_power(sunlight_intensity) / 60)
                self.battery.charge(solar_input)
                total_charge += solar_input
            self.hourly_data.append((hour + 6, self.battery.charge_level))
        print(f"Battery trickle charged by {total_charge:.2f} Wh over {duration_hours} hours")


# In[11]:


def regenerative_braking(self, speed, terrain):
    if terrain == "Downhill":
        recovered_energy = (self.motor_power * speed * 0.05)
        self.battery.charge(recovered_energy)
        print(f"Regenerative braking recovered {recovered_energy:.2f} Wh")


# In[12]:


def charge_solar(self, duration_hours):
    total_charge = 0
    for hour in range(duration_hours):
        sunlight_intensity = self.weather.get_sunlight(hour + 6)
        for minute in range(60):
            solar_input = (self.solar_panel.generate_power(sunlight_intensity) / 60)
            self.battery.charge(solar_input)
            total_charge += solar_input
        self.hourly_data.append((hour + 6, self.battery.charge_level))

    print(f"Battery trickle charged by {total_charge:.2f} Wh over {duration_hours} hours")


# In[13]:


def drive(self, distance, terrain):
    terrain_factor = {"Flat": 1.0, "Hill": 1.5, "Sandy": 1.8, "Rough": 2.0, "Downhill": 0.7}.get(terrain, 1.0)
    energy_needed = self.motor.energy_required(distance, terrain_factor)
    boost_needed = min(energy_needed * 0.2, self.supercapacitor.charge_level)

    self.supercapacitor.provide_boost(boost_needed)
    remaining_energy_needed = energy_needed - boost_needed
    total_available = self.battery.charge_level + self.supercapacitor.charge_level

    if total_available >= remaining_energy_needed:
        battery_ratio = self.battery.charge_level / total_available
        battery_usage = remaining_energy_needed * battery_ratio

        self.battery.discharge(battery_usage)
        self.total_energy_consumed += remaining_energy_needed
        self.total_distance_covered += distance
        self.terrain_energy_usage[terrain] += remaining_energy_needed

        wh_per_km = self.total_energy_consumed / self.total_distance_covered
        print(f"Energy Efficiency: {wh_per_km:.2f} Wh/km | Terrain: {terrain}")
    else:
        print("Warning: Not enough energy to complete trip!")


# In[14]:


# Define terrains, distances, and speeds (add speed for downhill)
terrains = ["Flat", "Hill", "Sandy", "Rough", "Downhill"]
distances = [10, 10, 10, 10, 10]  # Example distances per terrain (km)
speeds = [30, 25, 20, 15, 40]  # Example speeds per terrain (km/h)

# Create the tuktuk
motor = Motor(power_rating=5000, efficiency=0.85)
solar_tuktuk = SolarTukTuk(
    battery_capacity=5000,
    capacitor_capacity=500,
    motor=motor,
    kerb_weight=400,
    top_speed=80,
    panel_area=1.5,
    panel_efficiency=0.2
)

# Solar charge before trips
solar_tuktuk.charge_solar(duration_hours=6)

# Simulate trips and collect metrics
results = []
for terrain, distance, speed in zip(terrains, distances, speeds):
    before_energy = solar_tuktuk.battery.charge_level + solar_tuktuk.supercapacitor.charge_level

    # Track recovered energy for this trip
    recovered_energy = 0
    # Drive as usual
    solar_tuktuk.drive(distance=distance, terrain=terrain, speed=speed if terrain == "Downhill" else None)
    # If downhill, call regenerative braking and track recovered energy
    if terrain == "Downhill":
        # Regenerative braking is now called inside drive, but if not, call it here:
        # solar_tuktuk.regenerative_braking(speed, terrain)
        # Calculate recovered energy (assuming method adds to battery directly)
        # For tracking, you can re-calculate or store in the class as an attribute
        # Here, let's recalculate for reporting:
        recovered_energy = (motor.power_rating * speed * 0.05)

    after_energy = solar_tuktuk.battery.charge_level + solar_tuktuk.supercapacitor.charge_level
    energy_used = before_energy - after_energy - recovered_energy  # Subtract recovered energy to get net use

    results.append({
        "terrain": terrain,
        "distance_km": distance,
        "energy_used_Wh": energy_used,
        "recovered_Wh": recovered_energy,
        "net_energy_Wh": energy_used - recovered_energy,
        "efficiency_Wh_per_km": (energy_used - recovered_energy) / distance if distance > 0 else None
    })

# Convert to DataFrame for easier analysis
import pandas as pd
df = pd.DataFrame(results)

# Show the DataFrame
print(df)


# In[15]:


import ipywidgets as widgets
from ipywidgets import interact
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make sure your Motor, SolarTukTuk, ElectricTukTuk classes are defined above!

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
    cap_weight_per_Wh=0.04
):
    terrains = ["Flat", "Hill", "Sandy", "Rough", "Downhill"]
    speeds = [30, 25, 20, 15, 40]

    # Calculate kerb weights
    solar_kerb_weight = base_kerb_weight + (panel_area * panel_weight_per_m2) + (capacitor_capacity * cap_weight_per_Wh)
    electric_kerb_weight = base_kerb_weight

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
    electric_tuktuk = ElectricTukTuk(
        battery_capacity=battery_capacity,
        motor=motor,
        kerb_weight=electric_kerb_weight,
        top_speed=80
    )
    if trickle_charge:
        solar_tuktuk.charge_solar(duration_hours=solar_hours)

    results = []
    for terrain, speed in zip(terrains, speeds):
        distance = distance_per_terrain

        # Solar
        before_energy_solar = solar_tuktuk.battery.charge_level + solar_tuktuk.supercapacitor.charge_level
        energy_needed_solar = motor.energy_required(
            distance, 
            {"Flat": 1.0, "Hill": 1.5, "Sandy": 1.8, "Rough": 2.0, "Downhill": 0.7}[terrain],
            kerb_weight=solar_kerb_weight
        )
        recovered_solar = min(energy_needed_solar * 0.1, motor.power_rating * speed * 0.005) if terrain == "Downhill" else 0
        solar_tuktuk.drive(distance=distance, terrain=terrain, speed=speed if terrain == "Downhill" else None)
        after_energy_solar = solar_tuktuk.battery.charge_level + solar_tuktuk.supercapacitor.charge_level
        net_energy_used_solar = before_energy_solar - after_energy_solar - recovered_solar

        # Electric
        before_energy_electric = electric_tuktuk.battery.charge_level
        energy_needed_electric = motor.energy_required(
            distance, 
            {"Flat": 1.0, "Hill": 1.5, "Sandy": 1.8, "Rough": 2.0, "Downhill": 0.7}[terrain],
            kerb_weight=electric_kerb_weight
        )
        recovered_electric = min(energy_needed_electric * 0.1, motor.power_rating * speed * 0.005) if terrain == "Downhill" else 0
        electric_tuktuk.drive(distance=distance, terrain=terrain, speed=speed if terrain == "Downhill" else None)
        after_energy_electric = electric_tuktuk.battery.charge_level
        net_energy_used_electric = before_energy_electric - after_energy_electric - recovered_electric

        # Range calculation (fresh vehicles to avoid depletion)
        def calc_range(vehicle, terrain, speed, kerb_weight, trickle_charge, solar_hours=6):
            if trickle_charge and hasattr(vehicle, 'charge_solar'):
                vehicle.charge_solar(duration_hours=solar_hours)
            terrain_factor = {"Flat": 1.0, "Hill": 1.5, "Sandy": 1.8, "Rough": 2.0, "Downhill": 0.7}[terrain]
            energy_per_km = vehicle.motor.energy_required(1, terrain_factor, kerb_weight=kerb_weight)
            if terrain == "Downhill":
                recovered_per_km = min(energy_per_km * 0.1, vehicle.motor.power_rating * speed * 0.005 / 1)
            else:
                recovered_per_km = 0
            net_energy_per_km = max(energy_per_km - recovered_per_km, 0.01)
            if hasattr(vehicle, 'supercapacitor'):
                available_energy = vehicle.battery.charge_level + vehicle.supercapacitor.charge_level
            else:
                available_energy = vehicle.battery.charge_level
            return available_energy / net_energy_per_km

        solar_vehicle = SolarTukTuk(
            battery_capacity=battery_capacity,
            capacitor_capacity=capacitor_capacity,
            motor=motor,
            kerb_weight=solar_kerb_weight,
            top_speed=80,
            panel_area=panel_area,
            panel_efficiency=panel_efficiency
        )
        electric_vehicle = ElectricTukTuk(
            battery_capacity=battery_capacity,
            motor=motor,
            kerb_weight=electric_kerb_weight,
            top_speed=80
        )
        solar_range = calc_range(solar_vehicle, terrain, speed, solar_kerb_weight, trickle_charge, solar_hours)
        electric_range = calc_range(electric_vehicle, terrain, speed, electric_kerb_weight, False, 0)

        results.append({
            "terrain": terrain,
            "distance_km": distance,
            "solar_eff_Wh_per_km": net_energy_used_solar / distance if distance > 0 else None,
            "electric_eff_Wh_per_km": net_energy_used_electric / distance if distance > 0 else None,
            "solar_range_km": solar_range,
            "electric_range_km": electric_range
        })

    df = pd.DataFrame(results)
    # Calculate running cost per km
    df['solar_cost_per_km'] = df['solar_eff_Wh_per_km'] / 1000 * 0  # Solar running cost (USD)
    df['electric_cost_per_km'] = df['electric_eff_Wh_per_km'] / 1000 * grid_cost_per_kwh

    # Annual running cost (per terrain)
    df['solar_annual_cost'] = df['solar_cost_per_km'] * annual_km
    df['electric_annual_cost'] = df['electric_cost_per_km'] * annual_km

    # Total cost over one year (initial + running)
    df['solar_total_cost_year'] = initial_investment_solar + df['solar_annual_cost']
    df['electric_total_cost_year'] = initial_investment_electric + df['electric_annual_cost']

    # --- Energy Efficiency Plot with Weight Breakdown ---
    fig_eff, ax_eff = plt.subplots(figsize=(12, 6))
    
    # Calculate weight components
    base_weight = base_kerb_weight
    panel_weight = panel_area * panel_weight_per_m2
    capacitor_weight = capacitor_capacity * cap_weight_per_Wh
    
    # Create proportional breakdown for SolarTukTuk
    solar_eff = df['solar_eff_Wh_per_km'].values
    electric_eff = df['electric_eff_Wh_per_km'].values
    
    total_solar_weight = base_weight + panel_weight + capacitor_weight
    solar_eff_base = solar_eff * (base_weight / total_solar_weight)
    solar_eff_panel = solar_eff * (panel_weight / total_solar_weight)
    solar_eff_cap = solar_eff * (capacitor_weight / total_solar_weight)
    
    # Plot stacked bars for SolarTukTuk
    bottom = np.zeros(len(df['terrain']))
    p1 = ax_eff.bar(df['terrain'], solar_eff_base, width=0.4, label=f'Solar (Base: {base_weight}kg)', align='center', color='#1f77b4')
    bottom += solar_eff_base
    p2 = ax_eff.bar(df['terrain'], solar_eff_panel, width=0.4, label=f'Solar (Panel: {panel_weight:.1f}kg)', align='center', bottom=bottom, color='#aec7e8')
    bottom += solar_eff_panel
    p3 = ax_eff.bar(df['terrain'], solar_eff_cap, width=0.4, label=f'Solar (Cap: {capacitor_weight:.1f}kg)', align='center', bottom=bottom, color='#4b78c9')
    
    # Plot ElectricTukTuk
    p4 = ax_eff.bar(df['terrain'], electric_eff, width=0.4, label=f'Electric (Base: {base_weight}kg)', align='edge', color='#ff7f0e')
    
    ax_eff.set_xlabel("Terrain Type")
    ax_eff.set_ylabel("Energy Efficiency (Wh/km)")
    ax_eff.set_title(
        f"Energy Efficiency Breakdown by Weight Components\n"
        f"SolarTukTuk Total Weight: {total_solar_weight:.1f}kg vs ElectricTukTuk: {base_weight}kg"
    )
    ax_eff.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_eff.grid(True, linestyle='--', alpha=0.7)
    ax_eff.set_ylim(0, max(df['solar_eff_Wh_per_km'].max(), df['electric_eff_Wh_per_km'].max()) * 1.2)
    plt.tight_layout()

    # --- Range Plot ---
    fig_range, ax_range = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = range(len(df['terrain']))
    ax_range.bar([i - bar_width/2 for i in x], df['solar_range_km'], width=bar_width, label='SolarTukTuk', color='#1f77b4')
    ax_range.bar([i + bar_width/2 for i in x], df['electric_range_km'], width=bar_width, label='ElectricTukTuk', color='#ff7f0e')
    ax_range.set_xticks(list(x))
    ax_range.set_xticklabels(df['terrain'])
    ax_range.set_xlabel("Terrain Type")
    ax_range.set_ylabel("Estimated Range (km)")
    ax_range.set_title(f"Vehicle Range by Terrain\n(Trickle Charging: {'ON' if trickle_charge else 'OFF'})")
    ax_range.legend()
    ax_range.grid(True, linestyle='--', alpha=0.7)
    ax_range.set_ylim(0, max(df['solar_range_km'].max(), df['electric_range_km'].max()) * 1.2)
    plt.tight_layout()

    # --- Cost Comparison Plot ---
    fig_cost, ax_cost = plt.subplots(figsize=(10, 6))
    ax_cost.bar([i - bar_width/2 for i in x], df['solar_total_cost_year'], width=bar_width, label='SolarTukTuk', color='#1f77b4')
    ax_cost.bar([i + bar_width/2 for i in x], df['electric_total_cost_year'], width=bar_width, label='ElectricTukTuk', color='#ff7f0e')
    ax_cost.set_xticks(list(x))
    ax_cost.set_xticklabels(df['terrain'])
    ax_cost.set_xlabel("Terrain Type")
    ax_cost.set_ylabel("Total Cost (USD)")
    ax_cost.set_title(f"1st Year Total Cost Comparison\n({annual_km} km annual distance)")
    ax_cost.legend()
    ax_cost.grid(True, linestyle='--', alpha=0.7)
    ax_cost.set_ylim(0, max(df['solar_total_cost_year'].max(), df['electric_total_cost_year'].max()) * 1.1)
    plt.tight_layout()

    # --- DataFrames ---
    df_efficiency = df[['terrain', 'solar_eff_Wh_per_km', 'electric_eff_Wh_per_km']]
    df_efficiency.columns = ['Terrain', 'SolarTukTuk (Wh/km)', 'ElectricTukTuk (Wh/km)']
    df_efficiency = df_efficiency.round(2)

    df_range = df[['terrain', 'solar_range_km', 'electric_range_km']]
    df_range.columns = ['Terrain', 'SolarTukTuk Range (km)', 'ElectricTukTuk Range (km)']
    df_range = df_range.round(1)

    df_cost = df[['terrain', 'solar_cost_per_km', 'electric_cost_per_km', 'solar_total_cost_year', 'electric_total_cost_year']]
    df_cost.columns = [
        'Terrain', 
        'SolarTukTuk Cost/km (USD)', 
        'ElectricTukTuk Cost/km (USD)', 
        'SolarTukTuk Total 1st Year (USD)', 
        'ElectricTukTuk Total 1st Year (USD)'
    ]
    df_cost.iloc[:, 1:] = df_cost.iloc[:, 1:].round(2)

    return {
        'efficiency_plot': fig_eff,
        'range_plot': fig_range,
        'cost_plot': fig_cost,
        'df_efficiency': df_efficiency,
        'df_range': df_range,
        'df_cost': df_cost,
        'weight_breakdown': {
            'base_weight': base_weight,
            'panel_weight': panel_weight,
            'capacitor_weight': capacitor_weight,
            'total_solar_weight': total_solar_weight
        }
    }

# Create interactive widgets
interactive_simulation = interact(
    run_simulation,
    battery_capacity=widgets.IntSlider(min=1000, max=10000, step=100, value=5000),
    capacitor_capacity=widgets.IntSlider(min=0, max=2000, step=50, value=500),
    motor_power=widgets.IntSlider(min=1000, max=10000, step=100, value=5000),
    motor_efficiency=widgets.FloatSlider(min=0.5, max=0.95, step=0.05, value=0.85),
    panel_area=widgets.FloatSlider(min=0.5, max=3, step=0.1, value=1.5),
    panel_efficiency=widgets.FloatSlider(min=0.1, max=0.3, step=0.01, value=0.2),
    trickle_charge=widgets.Checkbox(value=True),
    solar_hours=widgets.IntSlider(min=1, max=12, step=1, value=6),
    distance_per_terrain=widgets.IntSlider(min=1, max=50, step=1, value=10),
    base_kerb_weight=widgets.IntSlider(min=200, max=600, step=10, value=400),
    panel_weight_per_m2=widgets.FloatSlider(min=5, max=20, step=0.5, value=12.5),
    cap_weight_per_Wh=widgets.FloatSlider(min=0.01, max=0.1, step=0.005, value=0.04)
)