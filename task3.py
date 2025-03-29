import numpy as np
from data import get_fixed_data
from PriceProcess import price_model
from WindProcess import wind_model

#storing theta
class LinearValueFunction:
    def __init__(self, theta):
        self.theta = theta 

    def predict_single(self, x):
        x = np.array(x)
        return float(np.dot(self.theta, x))

def train_linear_value_function(states, target_values):
    X = []
    for lam, wind, y, s, _ in states:
        X.append([lam, wind, y, s])
    X = np.array(X)
    y = np.array(target_values)
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return LinearValueFunction(theta)

#sampling "I" states
#guys I'm treating egrid as another thing we have to sample randomically otherwise IDK how to compute the cost function
#should we use a policy to estimate it better? 
#maybe we should ask
def sample_representative_states(I, problem_data):
    states = []
    for _ in range(I):
        lam = np.random.uniform(0, 90)
        wind = np.random.uniform(0, 35)
        y = np.random.choice([0, 1])
        s = np.random.uniform(0, problem_data['hydrogen_capacity'])
        egrid = np.random.uniform(0, 10)
        states.append((lam, wind, y, s, egrid))
    return states

#sampling K exougenous states
#I simulated random values for previous wind and price and then used the next wind/price function to estimate next state
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
def compute_target_value(state, exo_future_samples, value_fn_next, problem_data, gamma=0.9):
    lam_t, wind_t, y_t, s_t, egrid = state
    Celzr = problem_data['electrolyzer_cost']
    immediate_cost = lam_t * egrid + Celzr * y_t
    future_values = []
    for lam_next, wind_next in exo_future_samples:
        x_next = [lam_next, wind_next, y_t, s_t]
        v_next = max(0, value_fn_next.predict_single(x_next))  
        future_values.append(v_next)
    expected_future = np.mean(future_values)
    return max(0, immediate_cost + gamma * expected_future)

#I picked out random values of I and K to see if it ran
problem_data = get_fixed_data()
T = 24  
I = 20  
K = 3  
gamma = 0.9

theta_by_t = {}
value_fn_next = LinearValueFunction(theta=np.zeros(4))  

#main loop, didn't define it as a function yet
for t in reversed(range(T)):
    states_t = sample_representative_states(I, problem_data)
    targets = []

    for state in states_t:
        lam_t, wind_t, *_ = state
        exo_samples = sample_exogenous_next_states(lam_t, wind_t, K, problem_data)
        V_target = compute_target_value(state, exo_samples, value_fn_next, problem_data, gamma)
        targets.append(V_target)

    value_fn_t = train_linear_value_function(states_t, targets)
    theta_by_t[t] = value_fn_t.theta
    value_fn_next = value_fn_t

    avg_value = np.mean(targets)
    print(f"  t = {t}: mean V_target = {avg_value:.2f}")
#to see what values where drawn
#should we have a correlation between pt egrid and lambda?
print(f"t = {t} | λ values: {[round(s[0], 2) for s in states_t]}")
print(f"t = {t} | egrid values: {[round(s[4], 2) for s in states_t]}")