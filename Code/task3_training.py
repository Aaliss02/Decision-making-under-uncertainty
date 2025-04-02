import numpy as np
from pyomo.environ import *
from data import get_fixed_data
from PriceProcess import price_model
from WindProcess import wind_model

I = 100
K = 100
gamma = 0.9
problemData = get_fixed_data()

theta = {}
with open("values.txt", "r") as file:
    for t, line in enumerate(file):
        values = list(map(float, line.strip().split()))
        theta[t] = values

def sample_representative_states(I):
    states = []
    for _ in range(I):
        lam = np.random.uniform(0, 50)
        wind = np.random.uniform(0, 50)
        y = np.random.choice([0, 1])
        s = np.random.uniform(0, problemData['hydrogen_capacity'])
        states.append([lam, wind, y, s])
    return states

def sample_exogenous_next_states(lam_t, wind_t, K):
    samples = []
    for _ in range(K):
        wind_t_minus1 = np.random.uniform(0, 50)    
        lam_t_minus1 = np.random.uniform(0, 50)
        wind_next = wind_model(wind_t, wind_t_minus1, problemData)
        lam_next = price_model(lam_t, lam_t_minus1, wind_next, problemData)
        samples.append([lam_next, wind_next])
    return samples

for t in reversed(range(problemData['num_timeslots'])):
    print(t)
    X = []
    y = []
    states = sample_representative_states(I)
    targets = [0 for _ in range(I)]
    i = 0
    for state in states:
        lam_t, wind_t, y_t, s_t = state
        exo_samples = sample_exogenous_next_states(lam_t, wind_t, K)

        model = ConcreteModel()
        model.on = Var(bounds=(0, 1), within=Binary)
        model.off = Var(bounds=(0, 1), within=Binary)
        model.egrid = Var(within=NonNegativeReals)
        model.eelzr = Var(bounds=(0, problemData['p2h_rate']/problemData['conversion_p2h']))
        model.h = Var(bounds=(0, problemData['h2p_rate']/problemData['conversion_h2p']))

        model.demand_constraint = Constraint(expr=model.egrid + problemData['conversion_h2p'] * model.h + wind_t - model.eelzr >= problemData['demand_schedule'][t])
        model.on_and_off_constraint = Constraint(expr=model.on + model.off <= 1)
        model.on_constraint = Constraint(expr=y_t + model.on <= 1)
        model.off_constraint = Constraint(expr=model.off <= y_t)
        model.electrolyzer_constraint = Constraint(expr=problemData['conversion_p2h'] * model.eelzr <= y_t * problemData['p2h_rate'])
        model.electrolyzer_consumption_constraint = Constraint(expr=model.eelzr <= wind_t)
        model.current_storage_constraint = Constraint(expr=model.h <= s_t)
        model.future_storage_contraint = Constraint(expr=s_t - model.h + problemData['conversion_p2h'] * model.eelzr <= problemData['hydrogen_capacity'])

        if t+1 < len(theta):
            future_value = (1/K) * gamma * sum(
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

        targets[i] = value(model.objective)
        X.append([wind_t, lam_t, y_t, s_t]) 
        y.append(targets[i]) 
        i = i + 1

    model = ConcreteModel()
    model.theta1 = Var()
    model.theta2 = Var()
    model.theta3 = Var()
    model.theta4 = Var()
    model.objective = Objective(expr=sum(
        (model.theta1 * X[i][0] + model.theta2 * X[i][1] + model.theta3 * X[i][2] + model.theta4 * X[i][3] - y[i])**2 for i in range(I)
    ), sense=minimize)
    # Create a solver
    solver = SolverFactory('gurobi')  
    
    # Solve the model
    solver.solve(model)
    theta[t] = [value(model.theta1), value(model.theta2), value(model.theta3), value(model.theta4)]
    print(theta[t])

with open("values.txt", "w") as file:
    for t in range(len(theta)):
        values = " ".join(map(str, theta[t]))
        file.write(values + "\n")