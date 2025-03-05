from data import get_fixed_data
from WindProcess import wind_model
from PriceProcess import price_model
from Clustering import clustering
from pyomo.environ import *

def make_decision_two_stage(N, previous_and_current_price, previous_and_current_wind, current_and_next_demand, y1, s1):

    problemData = get_fixed_data()

    #create model
    model = ConcreteModel()

    # Declare variables
    model.S = Set(initialize=range(N))
    model.y2 = Var(model.S, bounds=(0, 1), within=Binary, initialize=0)
    model.on1 = Var(bounds=(0, 1), within=Binary, initialize=0)
    model.on2 = Var(model.S, bounds=(0, 1), within=Binary, initialize=0)
    model.off1 = Var(bounds=(0, 1), within=Binary, initialize=0)
    model.off2 = Var(model.S, bounds=(0, 1), within=Binary, initialize=0)
    model.s2 = Var(model.S, bounds=(0, problemData['hydrogen_capacity']), within=NonNegativeReals, initialize=0)
    model.egrid1 = Var(within=NonNegativeReals)
    model.egrid2 = Var(model.S, within=NonNegativeReals)
    model.eelzr1 = Var(bounds=(0, problemData['p2h_rate']/problemData['conversion_p2h']))
    model.eelzr2 = Var(model.S, bounds=(0, problemData['p2h_rate']/problemData['conversion_p2h']))
    model.h1 = Var(bounds=(0, problemData['h2p_rate']/problemData['conversion_h2p']))
    model.h2 = Var(model.S, bounds=(0, problemData['h2p_rate']/problemData['conversion_h2p']))
    
    prob = 1/N

    next_wind = [wind_model(previous_and_current_wind[1], previous_and_current_wind[0], problemData) for s in range(N)]
    next_price = [price_model(previous_and_current_price[1], previous_and_current_price[0], next_wind[s], problemData) for s in range(N)]

    # Objective function: Minimization of cost
    model.cost = Objective(
        expr= (previous_and_current_price[1] * model.egrid1) + problemData['electrolyzer_cost'] * y1 + 
        prob * sum(
            next_price[s] * model.egrid2[s] + problemData['electrolyzer_cost'] * model.y2[s]
            for s in model.S
        ),
        sense=minimize
    )

# Constraints

    #Constraint on demand
    model.Demand1 = Constraint(expr=model.egrid1 + (problemData['conversion_h2p'] * model.h1) + previous_and_current_wind[1] - model.eelzr1 >= current_and_next_demand[0])
    model.Demand2 = ConstraintList()
    for s in model.S:
        model.Demand2.add(model.egrid2[s] + (problemData['conversion_h2p'] * model.h2[s]) + next_wind[s] - model.eelzr2[s] >= current_and_next_demand[1])

    #Constraint on hydrogen conversion
    model.HydrogenConversion1 = Constraint(expr=model.h1 * problemData['conversion_h2p'] <= problemData['h2p_rate'])
    model.HydrogenConversion2 = ConstraintList()
    for s in model.S:
        model.HydrogenConversion2.add(model.h2[s] * problemData['conversion_h2p'] <= problemData['h2p_rate'])

    #Constraint on state of electrolyzer
    model.Electrolyzer1 = Constraint(expr=problemData['conversion_p2h'] * model.eelzr1 <= y1 * problemData['p2h_rate'])
    model.Electrolyzer2 = ConstraintList()
    for s in model.S:
        model.Electrolyzer2.add(problemData['conversion_p2h'] * model.eelzr2[s] <= model.y2[s]* problemData['p2h_rate'])

    model.OnAndOff1 = Constraint(expr=model.on1 + model.off1 <= 1)
    model.OnAndOff2 = ConstraintList()
    for s in model.S:
        model.OnAndOff2.add(model.on2[s] + model.off2[s] <= 1)

    model.On1 = Constraint(expr=y1 + model.on1 <= 1)
    model.On2 = ConstraintList()
    for s in model.S:
        model.On2.add(model.y2[s] + model.on2[s] <= 1)

    model.Off1 = Constraint(expr=model.off1 <= y1)
    model.Off2 = ConstraintList()
    for s in model.S:
        model.Off2.add(model.off2[s] <= model.y2[s])

    def elzr_relation(model, s):
        return model.y2[s] == y1 + model.on1 - model.off1  
    model.ElectrolyzerStat = Constraint(model.S, rule=elzr_relation)

    model.ElectrolyzerConsumption1 = Constraint(expr=model.eelzr1 <= previous_and_current_wind[1])
    model.ElectrolyzerConsumption2 = ConstraintList()
    for s in model.S:
        model.ElectrolyzerConsumption2.add(model.eelzr2[s] <= next_wind[s])

    #Constraints on storage
    model.CurrentStorage1 = Constraint(expr=model.h1 <= s1)
    model.CurrentStorage2 = ConstraintList()
    for s in model.S:
        model.CurrentStorage2.add(model.h2[s] <= model.s2[s])

    def storage_relation(model, s):
        return model.s2[s] == s1 - model.h1 + problemData['conversion_p2h'] * model.eelzr1
    model.storage_constraint = Constraint(model.S, rule=storage_relation)


    # Create a solver
    solver = SolverFactory('gurobi')  # Make sure Gurobi is installed and properly configured

    # Solve the model
    solver.solve(model, tee=True)

    return model.egrid1, model.eelzr1, model.h1, model.on1, model.off1