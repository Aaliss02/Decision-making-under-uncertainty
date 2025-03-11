import numpy as np
from data import get_fixed_data
from WindProcess import wind_model
from PriceProcess import price_model
from Task2 import make_decision_two_stage
from SP_multi_stage import make_decision_multi_stage
from pyomo.environ import value
from Task0 import optimize
from dummy_policy import make_dummy_decision

problemData = get_fixed_data()

nb_exp = 5
Expers = np.arange(nb_exp)
sim_T = range(1, problemData['num_timeslots'] + 1)
nb_scen = 8
lookahead = 4
nb_branches = 2

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

y = {}
egrid = {}
eelzr = {}
h = {}
on = {}
off = {}
s = {}
policy_cost = np.full((nb_exp, problemData['num_timeslots']), 99999999)
policy_cost_at_experiment = np.full(nb_exp, 99999999)

for e in Expers:
    for tau in sim_T:
        lookahead = min(lookahead, problemData['num_timeslots'] - tau + 1)

        demand = [problemData['demand_schedule'][tau - 1 + t] for t in range(lookahead)]

        if tau == 1:
            y[(e, tau)] = 0
            s[(e, tau)] = 0
            previous_and_current_wind = [5, wind_trajectory[e][0]]
            previous_and_current_price = [30, price_trajectory[e][0]]
        else:
            y[(e, tau)] = value(y[(e, tau - 1)] + on[(e, tau - 1)] - off[(e, tau - 1)])
            s[(e, tau)] = value(s[(e, tau - 1)] - h[(e, tau - 1)] + problemData['conversion_p2h'] * eelzr[(e, tau - 1)])
            previous_and_current_wind = [wind_trajectory[e][tau - 2], wind_trajectory[e][tau - 1]]
            previous_and_current_price = [price_trajectory[e][tau - 2], price_trajectory[e][tau - 1]]

        (
            egrid[(e, tau)],
            eelzr[(e, tau)],
            h[(e, tau)],
            on[(e, tau)],
            off[(e, tau)],
        ) = make_decision_multi_stage(nb_branches, nb_scen, lookahead, previous_and_current_price, previous_and_current_wind, demand, y[(e, tau)], s[(e, tau)])

        policy_cost[e][tau - 1] = value(price_trajectory[e][tau - 1] * egrid[(e, tau)] + problemData['electrolyzer_cost'] * y[(e, tau)])

    policy_cost_at_experiment[e] = sum(policy_cost[e, tau - 1] for tau in sim_T)

FINAL_POLICY_COST = sum(policy_cost_at_experiment[e] for e in Expers) / nb_exp
print("THE FINAL POLICY EXPECTED COST IS", FINAL_POLICY_COST)