import numpy as np
from pyomo.environ import *
from data import get_fixed_data
from PriceProcess import price_model
from WindProcess import wind_model

problemData = get_fixed_data()
theta = {}
with open("values.txt", "r") as file:
    for t, line in enumerate(file):
        values = list(map(float, line.strip().split()))
        theta[t] = values

print(theta)

def sample_exogenous_next_states(lam_t, wind_t, K, data):
    samples = []
    for _ in range(K):
        wind_t_minus1 = np.random.uniform(0, 50)    
        lam_t_minus1 = np.random.uniform(0, 50)
        wind_next = wind_model(wind_t, wind_t_minus1, data)
        lam_next = price_model(lam_t, lam_t_minus1, wind_next, data)
        samples.append([lam_next, wind_next])
    return samples

def adp_policy(current_state, t, gamma, S):
    lam_t, wind_t, y_t, s_t = current_state

    exo_samples = sample_exogenous_next_states(lam_t, wind_t, S, problemData)

    model = ConcreteModel()
    model.on = Var(bounds=(0, 1), within=Binary)
    model.off = Var(bounds=(0, 1), within=Binary)
    model.egrid = Var(within=NonNegativeReals)
    model.eelzr = Var(bounds=(0, problemData['p2h_rate']/problemData['conversion_p2h']))
    model.h = Var(bounds=(0, problemData['h2p_rate']/problemData['conversion_h2p']))

    model.demand_constraint = Constraint(expr=model.egrid + problemData['conversion_h2p'] * model.h + wind_t - model.eelzr >= problemData['demand_schedule'][t-1])
    model.on_and_off_constraint = Constraint(expr=model.on + model.off <= 1)
    model.on_constraint = Constraint(expr=y_t + model.on <= 1)
    model.off_constraint = Constraint(expr=model.off <= y_t)
    model.electrolyzer_constraint = Constraint(expr=problemData['conversion_p2h'] * model.eelzr <= y_t * problemData['p2h_rate'])
    model.electrolyzer_consumption_constraint = Constraint(expr=model.eelzr <= wind_t)
    model.current_storage_constraint = Constraint(expr=model.h <= s_t)
    model.future_storage_contraint = Constraint(expr=s_t - model.h + problemData['conversion_p2h'] * model.eelzr <= problemData['hydrogen_capacity'])

    if t+1 < len(theta):
        future_value = (1/S) * gamma * sum(
            theta[t+1][0]*samp[0] + theta[t+1][1]*samp[1] + 
            theta[t+1][2]*(y_t + model.on - model.off) + theta[t+1][3]*(s_t - model.h + problemData['conversion_p2h'] * model.eelzr)
            for samp in exo_samples
        )
    else:
        future_value = 0 

    model.objective = Objective(expr= (lam_t * model.egrid + problemData['electrolyzer_cost'] * y_t + future_value), sense=minimize)

    # Create a solver
    solver = SolverFactory('gurobi')  
    
    # Solve the model
    solver.solve(model)

    return value(model.egrid), value(model.eelzr), value(model.h), value(model.on), value(model.off)

    