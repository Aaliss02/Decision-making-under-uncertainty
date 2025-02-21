import numpy as np

# Load required modules (assuming equivalent Python functions exist)
from policy_module import multistage_policy, dummy_policy
from problem_data import load_the_data
from simulation_experiments import simulation_experiments_creation
from feasibility_check import check_feasibility

# Load problem parameters
(number_of_warehouses, W, cost_miss, cost_tr, warehouse_capacities, 
 transport_capacities, initial_stock, number_of_sim_periods, 
 sim_T, demand_trajectory) = load_the_data()

# Create random experiments
number_of_experiments, Expers, Price_experiments = simulation_experiments_creation(
    number_of_warehouses, W, number_of_sim_periods
)

# Initialize decision variables and policy cost
x, send, receive, z, m = {}, {}, {}, {}, {}

policy_cost = np.ones((number_of_experiments, number_of_sim_periods)) * 99999999
policy_cost_at_experiment = np.ones(number_of_experiments) * 99999999

# Iterate over each experiment
for e in Expers:
    for tau in sim_T:
        # Set each warehouse's stock level
        current_stock = initial_stock if tau == 1 else z[(e, tau - 1)]

        # Observe current demands and prices
        current_demands = demand_trajectory[:, tau]
        current_prices = Price_experiments[e, tau]

        # Call policy to make a decision
        x[(e, tau)], send[(e, tau)], receive[(e, tau)], m[(e, tau)] = multistage_policy(
            number_of_sim_periods, tau, current_stock, current_prices
        )

        # Check feasibility of decisions
        successful = check_feasibility(
            x[(e, tau)], send[(e, tau)], receive[(e, tau)], m[(e, tau)], 
            current_stock, current_demands, warehouse_capacities, transport_capacities
        )

        # If not feasible, use dummy policy
        if successful == 0:
            print("DECISION DOES NOT MEET CONSTRAINTS. USING DUMMY POLICY.")
            x[(e, tau)], send[(e, tau)], receive[(e, tau)], m[(e, tau)] = dummy_policy(
                number_of_sim_periods, tau, current_stock, current_demands, current_prices
            )

        # Calculate policy cost
        policy_cost[e, tau] = sum(
            current_prices[w] * x[(e, tau)][w] for w in W
        ) + sum(cost_tr[w, q] * receive[(e, tau)][w, q] for w in W for q in W) + \
        cost_miss[w] * m[(e, tau)][w]

    policy_cost_at_experiment[e] = sum(policy_cost[e, tau] for tau in sim_T)

# Compute final policy cost
FINAL_POLICY_COST = sum(policy_cost_at_experiment[e] for e in Expers) / number_of_experiments
print("THE FINAL POLICY EXPECTED COST IS", FINAL_POLICY_COST)
