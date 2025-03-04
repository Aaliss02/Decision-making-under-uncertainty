from data import get_fixed_data
from WindProcess import wind_model
from PriceProcess import price_model
from Task0 import optimize
from Clustering import clustering
import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import *
from Non_anticipativity import Non_anticipativity

def make_decision_multi_stage(nb_branches, nb_scen, lookahead, previous_and_current_price, previous_and_current_wind, current_and_next_demand, y1, s1):

    problemData = get_fixed_data()

    #create model
    model = ConcreteModel()

    # Declare variables
    model.S = Set(initialize=range(nb_scen))
    model.T = Set(initialize=range(lookahead))
    
    model.y = Var(model.S, model.T, bounds=(0, 1), within=Binary, initialize=0)
    model.on = Var(model.S, model.T, bounds=(0, 1), within=Binary, initialize=0)
    model.off = Var(model.S, model.T, bounds=(0, 1), within=Binary, initialize=0)
    model.s = Var(model.S, model.T, bounds=(0, problemData['hydrogen_capacity']), within=NonNegativeReals, initialize=0)
    model.egrid = Var(model.S, model.T, within=NonNegativeReals)
    model.eelzr = Var(model.S, model.T, bounds=(0, problemData['p2h_rate']/problemData['conversion_p2h']))
    model.h = Var(model.S, model.T, bounds=(0, problemData['h2p_rate']/problemData['conversion_h2p']))

    wind = [[previous_and_current_wind[1] for _ in range(lookahead)] for _ in range(nb_scen)]
    price = [[previous_and_current_price[1] for _ in range(lookahead)] for _ in range(nb_scen)]
    prob = [[1 for _ in range(lookahead)] for _ in range(nb_scen)]
    scen = [[[previous_and_current_wind[1],previous_and_current_price[1], 1] for _ in range(lookahead)] for _ in range(nb_scen)]
    for t in range(1,lookahead):
        for s in range(nb_scen):
            if t == 1:
                scen[s][t] = 

    print("clust", clustering(nb_branches, previous_and_current_wind[1], previous_and_current_wind[0], previous_and_current_price[1], previous_and_current_price[0], problemData))
    next_wind = [wind_model(previous_and_current_wind[1], previous_and_current_wind[0], problemData) for s in range(nb_scen)]
    next_price = [price_model(previous_and_current_price[1], previous_and_current_price[0], next_wind[s], problemData) for s in range(nb_scen)]

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
    model.Demand = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.Demand.add(model.egrid[s, t] + (problemData['conversion_h2p'] * model.h[s,t]) + wind[s,t] - model.eelzr2[s,t] >= demand[s,t])

    #Constraint on hydrogen conversion
    model.HydrogenConversion = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.HydrogenConversion.add(model.h[s,t] * problemData['conversion_h2p'] <= problemData['h2p_rate'])

    #Constraint on state of electrolyzer
    model.Electrolyzer = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.Electrolyzer.add(problemData['conversion_p2h'] * model.eelzr[s,t] <= model.y[s,t]* problemData['p2h_rate'])

    model.OnAndOff = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.OnAndOff.add(model.on[s,t] + model.off[s,t] <= 1)

    model.On = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.On2.add(model.y[s,t] + model.on[s,t] <= 1)

    model.Off = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.Off.add(model.off[s,t] <= model.y,t[s,t])

    def elzr_relation(model, s):
        return model.y[s, t+1] == model.y[s,t] + model.on[s,t] - model.off[s,t] 
    model.ElectrolyzerStat = Constraint(model.S, rule=elzr_relation)

    model.ElectrolyzerConsumption = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.ElectrolyzerConsumption.add(model.eelzr[s,t] <= next_wind[s,t])

    #Constraints on storage
    model.CurrentStorage = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.CurrentStorage.add(model.h[s,t] <= model.s[s,t])

    def storage_relation(model, s):
        return model.s[s, t+1] == s[s,t] - model.h[s,t] + problemData['conversion_p2h'] * model.eelzr[s,t]
    model.storage_constraint = Constraint(model.S, rule=storage_relation)

    #Non-Anticipativity constraints:
    S = Non_anticipativity()

    model.NonAn = ConstraintList()
    for s in model.S:
        for t in model.T:
            for s_prime in S.get((s,t), set()): #please nathan check we love u
                model.NonAn.add(model.h[s,t] == model.h[s_prime,t])


    # Create a solver
    solver = SolverFactory('gurobi')  # Make sure Gurobi is installed and properly configured

    # Solve the model
    solver.solve(model, tee=True)
    
    return model.egrid, model.eelzr, model.h, model.on, model.off