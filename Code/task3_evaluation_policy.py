import numpy as np
from data import get_fixed_data
from WindProcess import wind_model
from PriceProcess import price_model
from task3_policy import adp_policy  # Import the ADP policy function
from Task0 import optimize  # Import the optimization function
from dummy_policy import make_dummy_decision  # Import the dummy decision function
import matplotlib.pyplot as plt


# Load problem data
problem_data = get_fixed_data()

# Simulation parameters
nb_exp = 15
nb_timeslots = problem_data['num_timeslots']
sim_T = range(1, nb_timeslots + 1)

# Initialize trajectories and decision variables
wind_trajectory = [[0 for _ in range(nb_timeslots)] for _ in range(nb_exp)]
price_trajectory = [[0 for _ in range(nb_timeslots)] for _ in range(nb_exp)]
y = np.zeros((nb_exp, nb_timeslots + 1))  # y[e][tau] is status at start of tau
egrid = np.zeros((nb_exp, nb_timeslots + 1))
eelzr = np.zeros((nb_exp, nb_timeslots + 1))
h = np.zeros((nb_exp, nb_timeslots + 1))
on = np.zeros((nb_exp, nb_timeslots + 1))
off = np.zeros((nb_exp, nb_timeslots + 1))
s = np.zeros((nb_exp, nb_timeslots + 1))  # s[e][tau] is storage at start of tau
policy_cost = np.zeros((nb_exp, nb_timeslots))
policy_cost_at_experiment = np.zeros(nb_exp)
y_dummy = np.zeros((nb_exp, problem_data['num_timeslots']+1))
egrid_dummy = np.zeros((nb_exp, problem_data['num_timeslots']+1))
eelzr_dummy = np.zeros((nb_exp, problem_data['num_timeslots']+1))
h_dummy = np.zeros((nb_exp, problem_data['num_timeslots']+1))
on_dummy = np.zeros((nb_exp, problem_data['num_timeslots']+1))
off_dummy = np.zeros((nb_exp, problem_data['num_timeslots']+1))
s_dummy = np.zeros((nb_exp, problem_data['num_timeslots']+1))
policy_cost_dummy = np.zeros((nb_exp, problem_data['num_timeslots']))
policy_cost_at_experiment_dummy = np.zeros(nb_exp)

# Generate wind and price trajectories
for e in range(nb_exp):
    wind_trajectory[e][0] = wind_model(5, 4, problem_data)
    wind_trajectory[e][1] = wind_model(wind_trajectory[e][0], 5, problem_data)
    for i in range(1, nb_timeslots - 1):
        wind_trajectory[e][i + 1] = wind_model(wind_trajectory[e][i], wind_trajectory[e][i - 1], problem_data)

    price_trajectory[e][0] = price_model(30, 28, wind_trajectory[e][0], problem_data)
    price_trajectory[e][1] = price_model(price_trajectory[e][0], 30, wind_trajectory[e][1], problem_data)
    for i in range(1, nb_timeslots - 1):
        price_trajectory[e][i + 1] = price_model(price_trajectory[e][i], price_trajectory[e][i - 1], wind_trajectory[e][i + 1], problem_data)

# Simulate policy for each experiment
for e in range(nb_exp):
    for tau in range(1, nb_timeslots + 1):
        
        lookahead = 6
        current_lookahead = min(lookahead, nb_timeslots - tau + 1)
        demand = problem_data['demand_schedule'][tau-1:tau+current_lookahead-1]

        if tau == 1:
            previous_and_current_wind = [5, wind_trajectory[e][0]]
            previous_and_current_price = [30, price_trajectory[e][0]]
        else:
            y_dummy[e][tau] = y_dummy[e][tau-1] + on_dummy[e][tau-1] - off_dummy[e][tau-1]
            s_dummy[e][tau] = s_dummy[e][tau-1] - h_dummy[e][tau-1] + problem_data['conversion_p2h'] * eelzr_dummy[e][tau-1]
            y[e][tau] = y[e][tau-1] + on[e][tau-1] - off[e][tau-1]
            s[e][tau] = s[e][tau-1] - h[e][tau-1] + problem_data['conversion_p2h'] * eelzr[e][tau-1]
            previous_and_current_wind = [wind_trajectory[e][tau - 2], wind_trajectory[e][tau - 1]]
            previous_and_current_price = [price_trajectory[e][tau - 2], price_trajectory[e][tau - 1]]

        current_state = (price_trajectory[e][tau-1], wind_trajectory[e][tau-1], y[e][tau], s[e][tau])
        
        (egrid_dummy[e][tau], eelzr_dummy[e][tau], h_dummy[e][tau], on_dummy[e][tau], 
         off_dummy[e][tau]) = make_dummy_decision(demand[0], previous_and_current_wind[1])
        
        res = adp_policy(current_state, tau, 0.9, 50)

        (egrid[e][tau], eelzr[e][tau], h[e][tau], on[e][tau],
         off[e][tau]) = res
        
        policy_cost_dummy[e, tau - 1] = price_trajectory[e][tau-1] * egrid_dummy[e][tau] + problem_data['electrolyzer_cost'] * y_dummy[e][tau]
        policy_cost[e, tau - 1] = price_trajectory[e][tau-1] * egrid[e][tau] + problem_data['electrolyzer_cost'] * y[e][tau]

    policy_cost_at_experiment[e] = np.sum(policy_cost[e])
    
    policy_cost_at_experiment_dummy[e] = np.sum(policy_cost_dummy[e])
    print(f"Experiment {e} completed with total cost {policy_cost_at_experiment[e]:.2f}")
    print(f"Experiment {e} completed with total cost dummy {policy_cost_at_experiment_dummy[e]:.2f}")
    results = optimize(wind_trajectory[e], price_trajectory[e])
    print(f"Experiment {e} completed with total cost OIH {results['cost']:.2f}")

    if results['cost'] > policy_cost_at_experiment[e]:
        times = range(problem_data['num_timeslots'])

        plt.figure(figsize=(14, 12))

        plt.subplot(8, 1, 1)
        plt.plot(times, wind_trajectory[e], label="Wind Power", color="blue")
        plt.ylabel("Wind Power")
        plt.legend()

        plt.subplot(8, 1, 2)
        plt.plot(times, problem_data['demand_schedule'], label="Demand Schedule", color="orange")
        plt.ylabel("Demand")
        plt.legend()

        plt.subplot(8, 1, 3)
        plt.step(times, results['electrolyzer_status'], label="OIH El. Status", color="red", where="post")
        plt.step(times, y[e][1:], label="ADP El. Status", color="red", linestyle="dashed", where="post")
        plt.ylabel("El. Status")
        plt.legend()

        plt.subplot(8, 1, 4)
        plt.plot(times, results['hydrogen_storage_level'], label="OIH Hydrogen Level", color="green")
        plt.plot(times, s[e][1:], label="ADP Hydrogen Level", color="green", linestyle="dashed")
        plt.ylabel("Hydr. Level")
        plt.legend()

        plt.subplot(8, 1, 5)
        plt.plot(times, results['power_to_hydrogen'], label="OIH p2h", color="orange")
        plt.plot(times, eelzr[e][1:], label="ADP p2h", color="orange", linestyle="dashed")
        plt.ylabel("p2h")
        plt.legend()

        plt.subplot(8, 1, 6)
        plt.plot(times, results['hydrogen_to_power'], label="OIH h2p", color="blue")
        plt.plot(times, h[e][1:], label="ADP h2p", color="blue", linestyle="dashed")
        plt.ylabel("h2p")
        plt.legend()

        plt.subplot(8, 1, 7)
        plt.plot(times, results['grid_power'], label="OIH Grid Power", color="green")
        plt.plot(times, egrid[e][1:], label="ADP Grid Power", color="green", linestyle="dashed")
        plt.ylabel("Grid Power")
        plt.legend()

        plt.subplot(8, 1, 8)
        plt.plot(times, price_trajectory[e], label="Price", color="red")
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.legend()

        plt.tight_layout()
        plt.show()



# Compute final policy expected cost
FINAL_POLICY_COST = np.mean(policy_cost_at_experiment)
print(f"THE FINAL POLICY EXPECTED COST IS {FINAL_POLICY_COST:.2f}")

FINAL_POLICY_COST_dummy = np.mean(policy_cost_at_experiment_dummy)
print("THE FINAL DUMMY POLICY EXPECTED COST IS", FINAL_POLICY_COST_dummy)