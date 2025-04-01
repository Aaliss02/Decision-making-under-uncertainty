import numpy as np
from data import get_fixed_data
from WindProcess import wind_model
from PriceProcess import price_model
from task3_step2 import adp_policy  # Import the ADP policy function

# Load problem data
problem_data = get_fixed_data()

# Simulation parameters
nb_exp = 5
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
    y[e][0] = 0  # Initial electrolyzer status
    s[e][0] = 0  # Initial hydrogen storage
    for tau in range(1, nb_timeslots + 1):
        current_state = (price_trajectory[e][tau - 1], wind_trajectory[e][tau - 1], y[e][tau - 1], s[e][tau - 1])
        decisions = adp_policy(current_state, tau - 1, np.load('theta_by_t.npy', allow_pickle=True).item(), problem_data)

        if decisions:
            egrid[e][tau] = decisions['grid_power']
            eelzr[e][tau] = decisions['power_to_hydrogen']
            h[e][tau] = decisions['hydrogen_to_power']
            on[e][tau] = decisions['on_status']
            off[e][tau] = decisions['off_status']
            y[e][tau] = decisions['electrolyzer_status']  # y_tau after decisions
            s[e][tau] = decisions['hydrogen_storage_level']  # s_tau after decisions
            policy_cost[e, tau - 1] = decisions['cost']
        else:
            print(f"Experiment {e}, tau {tau}: No optimal solution found")
            # Default decisions in case of infeasibility
            egrid[e][tau] = 0
            eelzr[e][tau] = 0
            h[e][tau] = 0
            on[e][tau] = 0
            off[e][tau] = 0
            y[e][tau] = y[e][tau - 1]
            s[e][tau] = s[e][tau - 1]
            policy_cost[e, tau - 1] = 0  

    policy_cost_at_experiment[e] = np.sum(policy_cost[e])
    print(f"Experiment {e} completed with total cost {policy_cost_at_experiment[e]:.2f}")

# Compute final policy expected cost
FINAL_POLICY_COST = np.mean(policy_cost_at_experiment)
print(f"THE FINAL POLICY EXPECTED COST IS {FINAL_POLICY_COST:.2f}")