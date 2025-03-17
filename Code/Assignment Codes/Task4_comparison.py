import numpy as np
from data import get_fixed_data
from WindProcess import wind_model
from PriceProcess import price_model
from SP_multi_stage import make_decision_multi_stage
from pyomo.environ import value
from Task0 import optimize
from dummy_policy import make_dummy_decision
import matplotlib.pyplot as plt

problemData = get_fixed_data()

nb_exp = 20
Expers = np.arange(nb_exp)
sim_T = range(1, problemData['num_timeslots'] + 1)
nb_scen = 64
lookahead = 4
nb_branches = 4

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

y_dummy = {}
egrid_dummy = {}
eelzr_dummy = {}
h_dummy = {}
on_dummy = {}
off_dummy = {}
s_dummy = {}
policy_cost_dummy = np.full((nb_exp, problemData['num_timeslots']), 99999999)
policy_cost_at_experiment_dummy = np.full(nb_exp, 99999999)
y_multi_stage = {}
egrid_multi_stage = {}
eelzr_multi_stage = {}
h_multi_stage = {}
on_multi_stage = {}
off_multi_stage = {}
s_multi_stage = {}
policy_cost_multi_stage = np.full((nb_exp, problemData['num_timeslots']), 99999999)
policy_cost_at_experiment_multi_stage = np.full(nb_exp, 99999999)
y_expected_value = {}
egrid_expected_value = {}
eelzr_expected_value = {}
h_expected_value = {}
on_expected_value = {}
off_expected_value = {}
s_expected_value = {}
policy_cost_expected_value = np.full((nb_exp, problemData['num_timeslots']), 99999999)
policy_cost_at_experiment_expected_value = np.full(nb_exp, 99999999)
policy_cost_at_experiment_oih = np.full(nb_exp, 99999999)

for e in Expers:
    for tau in sim_T:
        lookahead = min(lookahead, problemData['num_timeslots'] - tau + 1)

        demand = [problemData['demand_schedule'][tau - 1 + t] for t in range(lookahead)]

        if tau == 1:
            y_dummy[(e, tau)] = 0
            s_dummy[(e, tau)] = 0
            y_multi_stage[(e, tau)] = 0
            s_multi_stage[(e, tau)] = 0
            y_expected_value[(e, tau)] = 0
            s_expected_value[(e, tau)] = 0
            previous_and_current_wind = [5, wind_trajectory[e][0]]
            previous_and_current_price = [30, price_trajectory[e][0]]
        else:
            y_dummy[(e, tau)] = y_dummy[(e, tau - 1)] + on_dummy[(e, tau - 1)] - off_dummy[(e, tau - 1)]
            s_dummy[(e, tau)] = s_dummy[(e, tau - 1)] - h_dummy[(e, tau - 1)] + problemData['conversion_p2h'] * eelzr_dummy[(e, tau - 1)]
            y_multi_stage[(e, tau)] = value(y_multi_stage[(e, tau - 1)] + on_multi_stage[(e, tau - 1)] - off_multi_stage[(e, tau - 1)])
            s_multi_stage[(e, tau)] = value(s_multi_stage[(e, tau - 1)] - h_multi_stage[(e, tau - 1)] + problemData['conversion_p2h'] * eelzr_multi_stage[(e, tau - 1)])
            y_expected_value[(e, tau)] = value(y_expected_value[(e, tau - 1)] + on_expected_value[(e, tau - 1)] - off_expected_value[(e, tau - 1)])
            s_expected_value[(e, tau)] = value(s_expected_value[(e, tau - 1)] - h_expected_value[(e, tau - 1)] + problemData['conversion_p2h'] * eelzr_expected_value[(e, tau - 1)])
            previous_and_current_wind = [wind_trajectory[e][tau - 2], wind_trajectory[e][tau - 1]]
            previous_and_current_price = [price_trajectory[e][tau - 2], price_trajectory[e][tau - 1]]

        (
            egrid_dummy[(e, tau)],
            eelzr_dummy[(e, tau)],
            h_dummy[(e, tau)],
            on_dummy[(e, tau)],
            off_dummy[(e, tau)],
        ) = make_dummy_decision(demand[0], previous_and_current_wind[1])

        (
            egrid_multi_stage[(e, tau)],
            eelzr_multi_stage[(e, tau)],
            h_multi_stage[(e, tau)],
            on_multi_stage[(e, tau)],
            off_multi_stage[(e, tau)],
        ) = make_decision_multi_stage(nb_branches, nb_scen, lookahead, previous_and_current_price, previous_and_current_wind, demand, y_multi_stage[(e, tau)], s_multi_stage[(e, tau)])

        (
            egrid_expected_value[(e, tau)],
            eelzr_expected_value[(e, tau)],
            h_expected_value[(e, tau)],
            on_expected_value[(e, tau)],
            off_expected_value[(e, tau)],
        ) = make_decision_multi_stage(1, 1, lookahead, previous_and_current_price, previous_and_current_wind, demand, y_expected_value[(e, tau)], s_expected_value[(e, tau)])

        policy_cost_dummy[e][tau - 1] = price_trajectory[e][tau - 1] * egrid_dummy[(e, tau)] + problemData['electrolyzer_cost'] * y_dummy[(e, tau)]
        policy_cost_multi_stage[e][tau - 1] = value(price_trajectory[e][tau - 1] * egrid_multi_stage[(e, tau)] + problemData['electrolyzer_cost'] * y_multi_stage[(e, tau)])
        policy_cost_expected_value[e][tau - 1] = value(price_trajectory[e][tau - 1] * egrid_expected_value[(e, tau)] + problemData['electrolyzer_cost'] * y_expected_value[(e, tau)])

    policy_cost_at_experiment_dummy[e] = sum(policy_cost_dummy[e, tau - 1] for tau in sim_T)
    policy_cost_at_experiment_multi_stage[e] = sum(policy_cost_multi_stage[e, tau - 1] for tau in sim_T)
    policy_cost_at_experiment_expected_value[e] = sum(policy_cost_expected_value[e, tau - 1] for tau in sim_T)
    res_oih = optimize(wind_trajectory[e], price_trajectory[e])
    policy_cost_at_experiment_oih[e] = res_oih['cost']
    print("Experiment ", e, " is done")

FINAL_POLICY_COST_dummy = sum(policy_cost_at_experiment_dummy[e] for e in Expers) / nb_exp
print("THE FINAL DUMMY POLICY EXPECTED COST IS", FINAL_POLICY_COST_dummy)
FINAL_POLICY_COST_multi_stage = sum(policy_cost_at_experiment_multi_stage[e] for e in Expers) / nb_exp
print("THE FINAL MULTI STAGE POLICY EXPECTED COST IS", FINAL_POLICY_COST_multi_stage)
FINAL_POLICY_COST_expected_value = sum(policy_cost_at_experiment_expected_value[e] for e in Expers) / nb_exp
print("THE FINAL EXPECTED VALUE POLICY EXPECTED COST IS", FINAL_POLICY_COST_expected_value)
FINAL_POLICY_COST_oih = sum(policy_cost_at_experiment_oih[e] for e in Expers) / nb_exp
print("THE FINAL OIH EXPECTED COST IS", FINAL_POLICY_COST_oih)

plt.figure(figsize=(10, 6))
plt.plot(Expers, policy_cost_at_experiment_dummy, label="Dummy Policy", marker='o')
plt.plot(Expers, policy_cost_at_experiment_multi_stage, label="Multi-Stage", marker='s')
plt.plot(Expers, policy_cost_at_experiment_expected_value, label="Expected Value", marker='^')
plt.plot(Expers, policy_cost_at_experiment_oih, label="OIH", marker='d')

# Labels and title
plt.xlabel("Experiment")
plt.ylabel("Policy Cost")
plt.title("Policy Cost at Each Experiment for Different Methods")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()