import numpy as np
from pyomo.environ import *
from data import get_fixed_data
from PriceProcess import price_model
from WindProcess import wind_model

# Load theta_by_t from the file
theta_by_t = np.load('theta_by_t.npy', allow_pickle=True).item()


class LinearValueFunction:
    def __init__(self, theta):
        self.theta = theta

    def predict_single(self, x):
        x = np.array(x)
        return float(np.dot(self.theta, x))

# Function to sample |S| next exogenous states (price and wind)
def sample_exogenous_next_states(lam_t, wind_t, S, data):
    samples = []
    for _ in range(S):
        wind_t_minus1 = np.random.uniform(0, 35)  # Same range as in the training code
        lam_t_minus1 = np.random.uniform(0, 90)   # Same range as in the training code
        wind_next = wind_model(wind_t, wind_t_minus1, data)
        lam_next = price_model(lam_t, lam_t_minus1, wind_next, data)
        samples.append((lam_next, wind_next))
    return samples

# Implementation of the ADP policy execution step
def adp_policy(current_state, tau, theta_by_t, problem_data, S=10, gamma=0.9): 
    # Unpack the current state
    lam_t, wind_t, y_t_minus1, s_t_minus1 = current_state

    # Sample |S| next exogenous states
    exo_samples = sample_exogenous_next_states(lam_t, wind_t, S, problem_data)

    # Single-period Pyomo model
    model = ConcreteModel()

    #  Variables for time tau (single period)
    model.yT = Var(bounds=(0, 1), within=Binary)
    model.onT = Var(bounds=(0, 1), within=Binary)
    model.offT = Var(bounds=(0, 1), within=Binary)
    model.sT = Var(bounds=(0, problem_data['hydrogen_capacity']), within=NonNegativeReals)
    model.egridT = Var(within=NonNegativeReals)
    model.eelzrT = Var(bounds=(0, problem_data['p2h_rate']/problem_data['conversion_p2h']))
    model.hT = Var(bounds=(0, problem_data['h2p_rate']/problem_data['conversion_h2p']))

    # Constraints 
    model.yT_constraint = Constraint(expr=model.yT == y_t_minus1 + model.onT - model.offT)
    model.sT_constraint = Constraint(expr=model.sT == s_t_minus1 - model.hT + problem_data['conversion_p2h'] * model.eelzrT)
    model.demand_constraint = Constraint(expr=model.egridT + problem_data['conversion_h2p'] * model.hT + wind_t - model.eelzrT >= problem_data['demand_schedule'][tau])
    model.on_and_off_constraint = Constraint(expr=model.onT + model.offT <= 1)
    model.on_constraint = Constraint(expr=model.yT + model.onT <= 1)
    model.off_constraint = Constraint(expr=model.offT <= model.yT)
    model.electrolyzer_constraint = Constraint(expr=problem_data['conversion_p2h'] * model.eelzrT <= model.yT * problem_data['p2h_rate'])
    model.hydrogen_conversion_constraint = Constraint(expr=model.hT * problem_data['conversion_h2p'] <= problem_data['h2p_rate'])
    model.electrolyzer_consumption_constraint = Constraint(expr=model.eelzrT <= wind_t)
    model.current_storage_constraint = Constraint(expr=model.hT <= model.sT)

    # Immediate cost
    immediate_cost = lam_t * model.egridT + problem_data['electrolyzer_cost'] * model.yT

    # Future value term 
    future_value = 0
    if tau + 1 in theta_by_t:  # If there’s a next period (tau < T-1)
        theta_next = theta_by_t[tau + 1]
        for lam_next, wind_next in exo_samples:
            # Compute V(z_{tau+1}, y_{tau+1}, s_{tau+1}, theta) symbolically
            v_next = (theta_next[0] * lam_next +
                      theta_next[1] * wind_next +
                      theta_next[2] * model.yT +
                      theta_next[3] * model.sT)
            future_value += v_next
        future_value = (gamma / S) * future_value
    # If tau = T-1, future_value remains 0 (no future cost)

    # Objective: Minimize immediate cost + discounted future cost
    model.objective = Objective(expr=immediate_cost + future_value, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')  
    results = solver.solve(model, tee=False)

    # Check if an optimal solution was found
    if results.solver.termination_condition == TerminationCondition.optimal:
        decisions = {
            'electrolyzer_status': value(model.yT),
            'on_status': value(model.onT),
            'off_status': value(model.offT),
            'hydrogen_storage_level': value(model.sT),
            'grid_power': value(model.egridT),
            'power_to_hydrogen': value(model.eelzrT),
            'hydrogen_to_power': value(model.hT),
            'cost': value(immediate_cost)
        }
        return decisions  # Return the decisions as a dictionary for here-and-now
    else:
        print(f"No optimal solution found at tau={tau}. Termination condition: {results.solver.termination_condition}")
        return {}  # Return empty dictionary if no solution