# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:30:51 2025

@author: geots
"""
from data import get_fixed_data
from pyomo.environ import *

def optimize(wind_trajectory, price_trajectory):
    problemData = get_fixed_data()

    # Create a model
    model = ConcreteModel()

    # Declare variables
    model.T = Set(initialize=range(problemData['num_timeslots']))
    model.yT = Var(model.T, bounds=(0, 1), within=Binary, initialize=0)
    model.onT = Var(model.T, bounds=(0, 1), within=Binary, initialize=0)
    model.offT = Var(model.T, bounds=(0, 1), within=Binary, initialize=0)
    model.sT = Var(model.T, bounds=(0, problemData['hydrogen_capacity']), within=NonNegativeReals, initialize=0)
    model.egridT = Var(model.T, within=NonNegativeReals)
    model.eelzrT = Var(model.T, bounds=(0, problemData['p2h_rate']/problemData['conversion_p2h']))
    model.hT = Var(model.T, bounds=(0, problemData['h2p_rate']/problemData['conversion_h2p']))

    # Objective function: Minimization of cost
    model.cost = Objective(
        expr=sum(
            price_trajectory[t] * model.egridT[t] +
            (problemData['electrolyzer_cost'] * model.yT[t])
            for t in model.T
        ),
        sense=minimize
    )

    # Constraints
    model.Demand = ConstraintList()
    for t in model.T:
        model.Demand.add(model.egridT[t] + problemData['conversion_h2p'] * model.hT[t] + wind_trajectory[t] - model.eelzrT[t] >= problemData['demand_schedule'][t])

    model.HydrogenConversion = ConstraintList()
    for t in model.T:
        model.HydrogenConversion.add(model.hT[t] * problemData['conversion_h2p'] <= problemData['h2p_rate'])

    model.Electrolyzer = ConstraintList()
    for t in model.T:
        model.Electrolyzer.add(problemData['conversion_p2h'] * model.eelzrT[t] <= model.yT[t] * problemData['p2h_rate'])

    model.OnAndOff = ConstraintList()
    for t in model.T:
        model.OnAndOff.add(model.onT[t] + model.offT[t] <= 1)

    model.On = ConstraintList()
    for t in model.T:
        model.On.add(model.yT[t] + model.onT[t] <= 1)

    model.Off = ConstraintList()
    for t in model.T:
        model.Off.add(model.offT[t] <= model.yT[t])

    def elzr_relation(model, t):
        if t > 0:
            return model.yT[t] == model.yT[t-1] + model.onT[t-1] - model.offT[t-1]
        return Constraint.Skip
    model.ElectrolyzerStat = Constraint(model.T, rule=elzr_relation)

    model.ElectrolyzerConsumption = ConstraintList()
    for t in model.T:
        model.ElectrolyzerConsumption.add(model.eelzrT[t] <= wind_trajectory[t])

    model.CurrentStorage = ConstraintList()
    for t in model.T:
        model.CurrentStorage.add(model.hT[t] <= model.sT[t])

    def storage_relation(model, t):
        if t > 0:
            return model.sT[t] == model.sT[t-1] - model.hT[t] + problemData['conversion_p2h'] * model.eelzrT[t]
        return Constraint.Skip
    model.storage_constraint = Constraint(model.T, rule=storage_relation)
    model.initial_storage = Constraint(expr=model.sT[0] == 0)

    # Create a solver
    solver = SolverFactory('gurobi')  
    
    # Solve the model
    results = solver.solve(model, tee=False)

    # Check if an optimal solution was found
    if results.solver.termination_condition == TerminationCondition.optimal:
        #print(" Optimal solution found")

        # Print out variable values and objective value
        '''print("Variable values:")
        for t in model.T:
            print(f"Time period {t}:")
            print(f"  yT[{t}]: {value(model.yT[t]):.3f}")
            print(f"  sT[{t}]: {value(model.sT[t]):.3f}")
            print(f"  egridT[{t}]: {value(model.egridT[t]):.3f}")
            print(f"  hT[{t}]: {value(model.hT[t]):.3f}")
            print(f"  eelzrT[{t}]: {value(model.eelzrT[t]):.3f}")
            print(f"  wind[{t}]: {wind_trajectory[t]:.3f}")
            print(f"  demand[{t}]: {problemData['demand_schedule'][t]:.3f}")
            print(f"cost : {price_trajectory[t]*value(model.egridT[t]) + problemData['electrolyzer_cost']* value(model.yT[t])}")'''
        #print(f"\nObjective value: {value(model.cost):.3f}\n")

        res = {
            'electrolyzer_status': [value(model.yT[t]) for t in model.T],
            'hydrogen_storage_level': [value(model.sT[t]) for t in model.T],
            'power_to_hydrogen': [value(model.eelzrT[t]) for t in model.T],
            'hydrogen_to_power': [value(model.hT[t]) for t in model.T],
            'grid_power': [value(model.egridT[t]) for t in model.T],
            'cost': value(model.cost)
        }

        return res  # 

    else:
        print(" No optimal solution found. Termination condition:", results.solver.termination_condition)
        return {} 

