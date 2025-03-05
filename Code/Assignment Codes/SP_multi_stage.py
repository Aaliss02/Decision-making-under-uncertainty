from data import get_fixed_data
from WindProcess import wind_model
from PriceProcess import price_model
from Clustering import clustering
from pyomo.environ import *
from Non_anticipativity import Non_anticipativity

def make_decision_multi_stage(nb_branches, nb_scen, lookahead, previous_and_current_price, previous_and_current_wind, demand, y1, s1):

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

    for t in range(lookahead - 1):
        clusters = []
        proba = []
        nb_clustering = nb_branches**t
        group_size = nb_scen // nb_clustering

        for i in range(nb_clustering):
            if t == 0:
                c, p = clustering(
                    nb_branches, previous_and_current_wind[1], previous_and_current_wind[0], 
                    previous_and_current_price[1], previous_and_current_price[0], problemData
                )
            else:
                c, p = clustering(
                    nb_branches, wind[i * group_size][t], wind[i * group_size][t - 1], 
                    price[i * group_size][t], price[i * group_size][t - 1], problemData
                )
            clusters.append(c)
            proba.append(p)

        for s in range(nb_scen):
            cluster_index = s // group_size
            branch_index = (s % group_size) // (group_size // nb_branches) 

            wind[s][t+1] = clusters[cluster_index][branch_index][0]
            price[s][t+1] = clusters[cluster_index][branch_index][1] 
            prob[s][t+1] = proba[cluster_index][branch_index] 

    next_wind = [wind_model(previous_and_current_wind[1], previous_and_current_wind[0], problemData) for s in range(nb_scen)]
    next_price = [price_model(previous_and_current_price[1], previous_and_current_price[0], next_wind[s], problemData) for s in range(nb_scen)]

    # Objective function: Minimization of cost
    model.cost = Objective(
        expr=sum(
            prob[s][t] * (price[s][t] * model.egrid[s, t] + problemData['electrolyzer_cost'] * model.y[s, t])
            for s in model.S for t in model.T
        ),
        sense=minimize
    )

# Constraints

    #Constraint on demand
    model.Demand = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.Demand.add(model.egrid[s, t] + (problemData['conversion_h2p'] * model.h[s,t]) + wind[s][t] - model.eelzr[s,t] >= demand[s][t])

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
            model.On.add(model.y[s,t] + model.on[s,t] <= 1)

    model.Off = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.Off.add(model.off[s,t] <= model.y[s,t])

    model.ElectrolyzerStat = ConstraintList()
    for s in model.S:
        for t in range(lookahead - 1):
            model.ElectrolyzerStat.add(
                model.y[s, t+1] == model.y[s, t] + model.on[s, t] - model.off[s, t]
            )

    model.ElectrolyzerConsumption = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.ElectrolyzerConsumption.add(model.eelzr[s, t] <= wind[s][t])

    #Constraints on storage
    model.CurrentStorage = ConstraintList()
    for s in model.S:
        for t in model.T:
            model.CurrentStorage.add(model.h[s,t] <= model.s[s,t])

    model.storage_constraint = ConstraintList()
    for s in model.S:
        for t in range(lookahead - 1):
            model.storage_constraint.add(
                model.s[s, t+1] == model.s[s, t] - model.h[s, t] + problemData['conversion_p2h'] * model.eelzr[s, t]
            )

    #Non-Anticipativity constraints:
    S = Non_anticipativity(nb_scen,lookahead,wind,price)

    model.NonAn = ConstraintList()
    for s in model.S:
        for t in model.T:
            for s_prime in S.get((s,t), set()): #please nathan check we love u
                model.NonAn.add(model.on[s, t] == model.on[s_prime, t])
                model.NonAn.add(model.off[s, t] == model.off[s_prime, t]) 
                model.NonAn.add(model.egrid[s, t] == model.egrid[s_prime, t])
                model.NonAn.add(model.eelzr[s, t] == model.eelzr[s_prime, t])
                model.NonAn.add(model.h[s, t] == model.h[s_prime, t])


    # Create a solver
    solver = SolverFactory('gurobi')  # Make sure Gurobi is installed and properly configured

    # Solve the model
    solver.solve(model, tee=True)
    
    return model.egrid[0, 0], model.eelzr[0, 0], model.h[0, 0], model.on[0, 0], model.off[0, 0]