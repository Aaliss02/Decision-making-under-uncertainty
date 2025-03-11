def make_dummy_decision(current_demand, current_wind):
    return max(current_demand - current_wind, 0), 0, 0, 0, 0