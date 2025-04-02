import numpy as np
from data import get_fixed_data
from WindProcess import wind_model
from PriceProcess import price_model
from SP_multi_stage import make_decision_multi_stage
from Task0 import optimize
from dummy_policy import make_dummy_decision
import matplotlib.pyplot as plt
from task3_policy import adp_policy

problemData = get_fixed_data()

nb_exp = 20
nb_timeslots = problemData['num_timeslots']
lookahead = [6, 4, 3, 2]
nb_branches = [2, 4, 6, 30]
nb_scen = [32, 64, 36, 30]

wind_trajectory = [[0 for _ in range(problemData['num_timeslots'])] for _ in range(nb_exp)]
for e in range(nb_exp):
    wind_trajectory[e][0] = wind_model(5, 4, problemData)
    wind_trajectory[e][1] = wind_model(wind_trajectory[e][0], 5, problemData)
    for i in range(1, problemData['num_timeslots'] - 1):
        wind_trajectory[e][i+1] = wind_model(wind_trajectory[e][i], wind_trajectory[e][i-1], problemData)

price_trajectory = [[0 for _ in range(problemData['num_timeslots'])] for _ in range(nb_exp)]
for e in range(nb_exp):
    price_trajectory[e][0] = price_model(30, 28, wind_trajectory[e][0], problemData)
    price_trajectory[e][1] = price_model(price_trajectory[e][0], 30, wind_trajectory[e][1], problemData)
    for i in range(1, problemData['num_timeslots'] - 1):
        price_trajectory[e][i+1] = price_model(price_trajectory[e][i], price_trajectory[e][i-1], wind_trajectory[e][i+1], problemData)

y_dummy = np.zeros((nb_exp, problemData['num_timeslots']+1))
egrid_dummy = np.zeros((nb_exp, problemData['num_timeslots']+1))
eelzr_dummy = np.zeros((nb_exp, problemData['num_timeslots']+1))
h_dummy = np.zeros((nb_exp, problemData['num_timeslots']+1))
on_dummy = np.zeros((nb_exp, problemData['num_timeslots']+1))
off_dummy = np.zeros((nb_exp, problemData['num_timeslots']+1))
s_dummy = np.zeros((nb_exp, problemData['num_timeslots']+1))
policy_cost_dummy = np.zeros((nb_exp, problemData['num_timeslots']))
policy_cost_at_experiment_dummy = np.zeros(nb_exp)
y_multi_stage = [np.zeros((nb_exp, problemData['num_timeslots']+1)) for _ in range(4)]
egrid_multi_stage = [np.zeros((nb_exp, problemData['num_timeslots']+1)) for _ in range(4)]
eelzr_multi_stage = [np.zeros((nb_exp, problemData['num_timeslots']+1)) for _ in range(4)]
h_multi_stage = [np.zeros((nb_exp, problemData['num_timeslots']+1)) for _ in range(4)]
on_multi_stage = [np.zeros((nb_exp, problemData['num_timeslots']+1)) for _ in range(4)]
off_multi_stage = [np.zeros((nb_exp, problemData['num_timeslots']+1)) for _ in range(4)]
s_multi_stage = [np.zeros((nb_exp, problemData['num_timeslots']+1)) for _ in range(4)]
policy_cost_multi_stage = [np.zeros((nb_exp, problemData['num_timeslots'])) for _ in range(4)]
policy_cost_at_experiment_multi_stage = [np.zeros(nb_exp) for _ in range(4)]
y_expected_value = np.zeros((nb_exp, problemData['num_timeslots']+1))
egrid_expected_value = np.zeros((nb_exp, problemData['num_timeslots']+1))
eelzr_expected_value = np.zeros((nb_exp, problemData['num_timeslots']+1))
h_expected_value = np.zeros((nb_exp, problemData['num_timeslots']+1))
on_expected_value = np.zeros((nb_exp, problemData['num_timeslots']+1))
off_expected_value = np.zeros((nb_exp, problemData['num_timeslots']+1))
s_expected_value = np.zeros((nb_exp, problemData['num_timeslots']+1))
policy_cost_expected_value = np.zeros((nb_exp, problemData['num_timeslots']))
policy_cost_at_experiment_expected_value = np.zeros(nb_exp)
y_adp = np.zeros((nb_exp, nb_timeslots + 1))
egrid_adp = np.zeros((nb_exp, nb_timeslots + 1))
eelzr_adp = np.zeros((nb_exp, nb_timeslots + 1))
h_adp = np.zeros((nb_exp, nb_timeslots + 1))
on_adp = np.zeros((nb_exp, nb_timeslots + 1))
off_adp = np.zeros((nb_exp, nb_timeslots + 1))
s_adp = np.zeros((nb_exp, nb_timeslots + 1))
policy_cost_adp = np.zeros((nb_exp, nb_timeslots))
policy_cost_at_experiment_adp = np.zeros(nb_exp)
policy_cost_at_experiment_oih = np.zeros(nb_exp)

for e in range(nb_exp):
    for tau in range(1, nb_timeslots+1):
        current_lookahead = min(lookahead[0], nb_timeslots - tau + 1)
        demand = problemData['demand_schedule'][tau-1:tau+current_lookahead-1]

        if tau == 1:
            previous_and_current_wind = [5, wind_trajectory[e][0]]
            previous_and_current_price = [30, price_trajectory[e][0]]
        else:
            y_dummy[e][tau] = y_dummy[e][tau-1] + on_dummy[e][tau-1] - off_dummy[e][tau-1]
            s_dummy[e][tau] = s_dummy[e][tau-1] - h_dummy[e][tau-1] + problemData['conversion_p2h'] * eelzr_dummy[e][tau-1]
            y_expected_value[e][tau] = y_expected_value[e][tau-1] + on_expected_value[e][tau-1] - off_expected_value[e][tau-1]
            s_expected_value[e][tau] = s_expected_value[e][tau-1] - h_expected_value[e][tau-1] + problemData['conversion_p2h'] * eelzr_expected_value[e][tau-1]
            for i in range(4):
                y_multi_stage[i][e][tau] = y_multi_stage[i][e][tau-1] + on_multi_stage[i][e][tau-1] - off_multi_stage[i][e][tau-1]
                s_multi_stage[i][e][tau] = s_multi_stage[i][e][tau-1] - h_multi_stage[i][e][tau-1] + problemData['conversion_p2h'] * eelzr_multi_stage[i][e][tau-1]
            y_adp[e][tau] = y_adp[e][tau-1] + on_adp[e][tau-1] - off_adp[e][tau-1]
            s_adp[e][tau] = s_adp[e][tau-1] - h_adp[e][tau-1] + problemData['conversion_p2h'] * eelzr_adp[e][tau-1]
            previous_and_current_wind = [wind_trajectory[e][tau - 2], wind_trajectory[e][tau - 1]]
            previous_and_current_price = [price_trajectory[e][tau - 2], price_trajectory[e][tau - 1]]

        current_state = (price_trajectory[e][tau-1], wind_trajectory[e][tau-1], y_adp[e][tau], s_adp[e][tau])

        (egrid_dummy[e][tau], eelzr_dummy[e][tau], h_dummy[e][tau], on_dummy[e][tau], 
         off_dummy[e][tau]) = make_dummy_decision(demand[0], previous_and_current_wind[1])
        
        (egrid_expected_value[e][tau], eelzr_expected_value[e][tau], h_expected_value[e][tau], on_expected_value[e][tau], 
         off_expected_value[e][tau]) = make_decision_multi_stage(
            1, 1, current_lookahead, previous_and_current_price, previous_and_current_wind, demand, y_expected_value[e][tau], s_expected_value[e][tau]
        )

        for i in range(4):
            current_lookahead = min(lookahead[i], nb_timeslots - tau + 1)
            demand = problemData['demand_schedule'][tau-1:tau+current_lookahead-1]
            (egrid_multi_stage[i][e][tau], eelzr_multi_stage[i][e][tau], h_multi_stage[i][e][tau], on_multi_stage[i][e][tau],
             off_multi_stage[i][e][tau]) = make_decision_multi_stage(
                nb_branches[i], nb_scen[i], current_lookahead, previous_and_current_price, previous_and_current_wind, demand, y_multi_stage[i][e][tau], s_multi_stage[i][e][tau]
            )

        (egrid_adp[e][tau], eelzr_adp[e][tau], h_adp[e][tau], on_adp[e][tau],
         off_adp[e][tau]) = adp_policy(current_state, tau, 0.9, 50)
        
        policy_cost_dummy[e, tau - 1] = price_trajectory[e][tau-1] * egrid_dummy[e][tau] + problemData['electrolyzer_cost'] * y_dummy[e][tau]
        for i in range(4):
            policy_cost_multi_stage[i][e, tau - 1] = price_trajectory[e][tau-1] * egrid_multi_stage[i][e][tau] + problemData['electrolyzer_cost'] * y_multi_stage[i][e][tau]
        policy_cost_expected_value[e, tau - 1] = price_trajectory[e][tau-1] * egrid_expected_value[e][tau] + problemData['electrolyzer_cost'] * y_expected_value[e][tau]
        policy_cost_adp[e, tau - 1] = price_trajectory[e][tau-1] * egrid_adp[e][tau] + problemData['electrolyzer_cost'] * y_adp[e][tau]

    policy_cost_at_experiment_oih[e] = optimize(wind_trajectory[e], price_trajectory[e])['cost']
    policy_cost_at_experiment_dummy[e] = np.sum(policy_cost_dummy[e])
    for i in range(4):
        policy_cost_at_experiment_multi_stage[i][e] = np.sum(policy_cost_multi_stage[i][e])
    policy_cost_at_experiment_expected_value[e] = np.sum(policy_cost_expected_value[e])
    policy_cost_at_experiment_adp[e] = np.sum(policy_cost_adp[e])
    print(f"Experiment {e} completed")

FINAL_POLICY_COST_dummy = np.mean(policy_cost_at_experiment_dummy)
print("THE FINAL DUMMY POLICY EXPECTED COST IS", FINAL_POLICY_COST_dummy)
FINAL_POLICY_COST_expected_value =  np.mean(policy_cost_at_experiment_expected_value)
print("THE FINAL EXPECTED VALUE POLICY EXPECTED COST IS", FINAL_POLICY_COST_expected_value)
for i in range(4):
    FINAL_POLICY_COST_multi_stage = np.mean(policy_cost_at_experiment_multi_stage[i])
    print(f"THE FINAL MULTI STAGE POLICY EXPECTED COST FOR {nb_branches[i]} BRANCHES AND A LOOKAHEAD OF {lookahead[i]} IS", FINAL_POLICY_COST_multi_stage)
FINAL_POLICY_COST_adp = np.mean(policy_cost_at_experiment_adp)
print(f"THE FINAL ADP POLICY EXPECTED COST IS {FINAL_POLICY_COST_adp}")
FINAL_POLICY_COST_oih = np.mean(policy_cost_at_experiment_oih)
print("THE FINAL OIH EXPECTED COST IS", FINAL_POLICY_COST_oih)

plt.figure(figsize=(10, 6))

colors = ['blue', 'green', 'red', 'magenta', 'orange', 'black', 'purple', 'cyan']
plt.plot(range(nb_exp), policy_cost_at_experiment_dummy, label="Dummy Policy", color=colors[0])
for i in range(4):
    plt.plot(range(nb_exp), policy_cost_at_experiment_multi_stage[i], 
             label=f"Multi-Stage {nb_branches[i]}B {lookahead[i]}L", color=colors[i + 1])
plt.plot(range(nb_exp), policy_cost_at_experiment_expected_value, label="Expected Value", color=colors[5])
plt.plot(range(nb_exp), policy_cost_at_experiment_oih, label="OIH", color=colors[6])
plt.plot(range(nb_exp), policy_cost_at_experiment_adp, label="ADP", color=colors[7])

# Labels and title
plt.xlabel("Experiment")
plt.ylabel("Policy Cost")
plt.title("Policy Cost at Each Experiment for Different Methods")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()