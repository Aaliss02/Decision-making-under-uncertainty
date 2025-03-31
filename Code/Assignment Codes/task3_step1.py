import numpy as np
from pyomo.environ import *
from data import get_fixed_data
from PriceProcess import price_model
from WindProcess import wind_model

# #storing theta
class LinearValueFunction:
    def __init__(self, theta):
        self.theta = theta 

    def predict_single(self, x):
        x = np.array(x)
        return float(np.dot(self.theta, x))

def train_linear_value_function(states, target_values):
    X = []
    for lam, wind, y, s in states:
        X.append([lam, wind, y, s])
    X = np.array(X)
    y = np.array(target_values)
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return LinearValueFunction(theta)

# Sample I representative states (here in my version I have excluded egrid and treat it as a decision variable)
def sample_representative_states(I, problem_data):
    states = []
    for _ in range(I):
        lam = np.random.uniform(0, 90)
        wind = np.random.uniform(0, 35)
        y = np.random.choice([0, 1])
        s = np.random.uniform(0, problem_data['hydrogen_capacity'])
        states.append((lam, wind, y, s))
    return states

#sampling K exougenous states
def sample_exogenous_next_states(lam_t, wind_t, K, data):
    samples = []
    for _ in range(K):
        wind_t_minus1 = np.random.uniform(0, 35)
        lam_t_minus1 = np.random.uniform(0, 90)
        wind_next = wind_model(wind_t, wind_t_minus1, data)
        lam_next = price_model(lam_t, lam_t_minus1, wind_next, data)
        samples.append((lam_next, wind_next))
    return samples

#target value
def compute_target_value(state, exo_future_samples, value_fn_next, problem_data, gamma=0.9, t=0):
    lam_t, wind_t, y_t, s_t = state  # Unpack the state tuple
    Celzr = problem_data['electrolyzer_cost']

    # Solve a single-period optimization problem to compute the immediate cost
    model = ConcreteModel()
    model.yT = Var(bounds=(0, 1), within=Binary, initialize=y_t)
    model.onT = Var(bounds=(0, 1), within=Binary)
    model.offT = Var(bounds=(0, 1), within=Binary)
    model.sT = Var(bounds=(0, problem_data['hydrogen_capacity']), within=NonNegativeReals, initialize=s_t)
    model.egridT = Var(within=NonNegativeReals)
    model.eelzrT = Var(bounds=(0, problem_data['p2h_rate']/problem_data['conversion_p2h']))
    model.hT = Var(bounds=(0, problem_data['h2p_rate']/problem_data['conversion_h2p']))
    model.yT_constraint = Constraint(expr=model.yT == y_t)  # Fix yT to the sampled value
    model.sT_constraint = Constraint(expr=model.sT == s_t - model.hT + problem_data['conversion_p2h'] * model.eelzrT)
    model.demand_constraint = Constraint(expr=model.egridT + problem_data['conversion_h2p'] * model.hT + wind_t - model.eelzrT >= problem_data['demand_schedule'][t])
    model.on_and_off_constraint = Constraint(expr=model.onT + model.offT <= 1)
    model.on_constraint = Constraint(expr=model.yT + model.onT <= 1)
    model.off_constraint = Constraint(expr=model.offT <= model.yT)
    model.electrolyzer_constraint = Constraint(expr=problem_data['conversion_p2h'] * model.eelzrT <= model.yT * problem_data['p2h_rate'])
    model.hydrogen_conversion_constraint = Constraint(expr=model.hT * problem_data['conversion_h2p'] <= problem_data['h2p_rate'])
    model.electrolyzer_consumption_constraint = Constraint(expr=model.eelzrT <= wind_t)
    model.current_storage_constraint = Constraint(expr=model.hT <= s_t)  # Corrected constraint
    immediate_cost = lam_t * model.egridT + problem_data['electrolyzer_cost'] * model.yT
    model.objective = Objective(expr=immediate_cost, sense=minimize)
    solver = SolverFactory('gurobi')
    results = solver.solve(model, tee=False)

    if results.solver.termination_condition != TerminationCondition.optimal:
        return float('inf')  # Return a large value if infeasible

    # Extract next state variables
    s_next = value(model.sT)  # Compute s_{t+1}
    if value(model.onT) > 0.5:
        y_next = 1            # Compute y_{t+1}
    elif value(model.offT) > 0.5:
        y_next = 0
    else:
        y_next = y_t

    immediate_cost = value(immediate_cost)

    # Compute the future value using next state
    future_values = []
    for lam_next, wind_next in exo_future_samples:
        x_next = [lam_next, wind_next, y_next, s_next]  # Updated state vector
        v_next = max(0, value_fn_next.predict_single(x_next))
        future_values.append(v_next)
    expected_future = np.mean(future_values)
    return max(0, immediate_cost + gamma * expected_future)

# Main training function with multiple backward passes
def main_train_value_function(problem_data, T, I, K, gamma, max_outer_iterations, convergence_threshold):
    theta_by_t = {}
    value_fn_next = LinearValueFunction(theta=np.zeros(4))
    prev_theta_by_t = None
    outer_iter = 0

    # Outer loop for multiple backward passes
    while outer_iter < max_outer_iterations:
        print(f"\nOuter Iteration {outer_iter + 1}/{max_outer_iterations}")
        
        # Backward pass
        for t in reversed(range(T)):
            states_t = sample_representative_states(I, problem_data)
            targets = []

            for state in states_t:
                lam_t, wind_t, *_ = state
                exo_samples = sample_exogenous_next_states(lam_t, wind_t, K, problem_data)
                V_target = compute_target_value(state, exo_samples, value_fn_next, problem_data, gamma, t)
                if V_target == float('inf'):
                    continue  # Skip infeasible states
                targets.append(V_target)

            if not targets:  # If all states were infeasible
                print(f"Warning: No feasible states at t={t}, skipping")
                continue

            value_fn_t = train_linear_value_function(states_t, targets)
            theta_by_t[t] = value_fn_t.theta
            value_fn_next = value_fn_t

            avg_value = np.mean(targets)
            print(f"  t = {t}: mean V_target = {avg_value:.2f}")

        # Check for convergence
        if prev_theta_by_t is not None:
            max_change = max(np.linalg.norm(theta_by_t[t] - prev_theta_by_t[t]) for t in theta_by_t)
            print(f"Max change in theta: {max_change:.6f}")
            if max_change < convergence_threshold:
                print("Converged: Stopping early")
                break

        # Copy current theta_by_t for the next iteration
        prev_theta_by_t = {t: theta_by_t[t].copy() for t in theta_by_t}
        outer_iter += 1

    return theta_by_t

# Run the training
if __name__ == "__main__":
    problem_data = get_fixed_data()
    T = 24
    I = 100  # Increased sample size I put 20 just to try the code and se if it runs without had to wait for long time, I tried also with 500 seems to be a good trade betwen accuracy and time
    K = 3   # Increased sample size, I put 5 just to try
    gamma = 0.9
    max_outer_iterations = 1  #I had the option to do several big loop through T but it just add noise to the theta, so we should keep it to 1
    convergence_threshold = 1e-3

    theta_by_t = main_train_value_function(problem_data, T, I, K, gamma, max_outer_iterations, convergence_threshold)

    # Save theta_by_t to a file
    np.save('theta_by_t.npy', theta_by_t)
    print("Saved theta_by_t to 'theta_by_t.npy'")