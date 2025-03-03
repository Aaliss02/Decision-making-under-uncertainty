# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 12:59:29 2024

@author: geots
"""

from data import get_fixed_data
from WindProcess import wind_model
from PriceProcess import price_model
from Task0 import optimize
import matplotlib.pyplot as plt

data = get_fixed_data()

initial_state = {'hydrogen': 0, 'electrolyzer_status': 0}

wind_trajectory = [0 for i in range(data['num_timeslots'])]
wind_trajectory[0] = wind_model(5,4,data)
wind_trajectory[1] = wind_model(wind_trajectory[0], 5, data)

for i in range(1, data['num_timeslots'] - 1):
    wind_trajectory[i+1] = wind_model(wind_trajectory[i], wind_trajectory[i-1], data)

price_trajectory = [0 for i in range(data['num_timeslots'])]
price_trajectory[0] = price_model(30,28,wind_trajectory[0], data)
price_trajectory[1] = price_model(price_trajectory[0], 30, wind_trajectory[1], data)

for i in range(1, data['num_timeslots'] - 1):
    price_trajectory[i+1] = price_model(price_trajectory[i], price_trajectory[i-1], wind_trajectory[i+1], data)

times = range(data['num_timeslots'])

results = optimize(wind_trajectory, price_trajectory)

# Plot results
plt.figure(figsize=(14, 10))


plt.subplot(8, 1, 1)
plt.plot(times, wind_trajectory, label="Wind Power", color="blue")
plt.ylabel("Wind Power")
plt.legend()

plt.subplot(8, 1, 2)
plt.plot(times, data['demand_schedule'], label="Demand Schedule", color="orange")
plt.ylabel("Demand")
plt.legend()

plt.subplot(8, 1, 3)
plt.step(times, results['electrolyzer_status'], label="Electrolyzer Status", color="red", where="post")
plt.ylabel("El. Status")
plt.legend()


plt.subplot(8, 1, 4)
plt.plot(times, results['hydrogen_storage_level'], label="Hydrogen Level", color="green")
plt.ylabel("Hydr. Level")
plt.legend()


plt.subplot(8, 1, 5)
plt.plot(times, results['power_to_hydrogen'], label="p2h", color="orange")
plt.ylabel("p2h")
plt.legend()


plt.subplot(8, 1, 6)
plt.plot(times, results['hydrogen_to_power'], label="h2p", color="blue")
plt.ylabel("h2p")
plt.legend()


plt.subplot(8, 1, 7)
plt.plot(times, results['grid_power'], label="Grid Power", color="green")
plt.ylabel("Grid Power")
plt.legend()

plt.subplot(8, 1, 8)
plt.plot(times, price_trajectory, label="price", color="red")
plt.ylabel("Price")
plt.xlabel("Time")
plt.legend()


plt.tight_layout()
plt.show()