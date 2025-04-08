import numpy as np

def feasibility_check(y, on, off, eelzr, h, egrid, s, wind, problemData, demand):
    C = problemData['hydrogen_capacity']
    P2H = problemData['p2h_rate']
    R_p2h = problemData['conversion_p2h']
    H2P = problemData['h2p_rate']
    R_h2p = problemData['conversion_h2p']
    next_s = s - h + R_p2h * eelzr
    epsilon = 1e-6

    if on not in (0,1) or off not in (0,1) or y + on - off not in (0,1):
        print("Non-binary on/off/y values")
        return False
    elif on + off > 1 + epsilon:
        print("Both on and off are 1")
        return False
    elif y + on > 1 + epsilon:
        print("Electrolyzer is already on")
        return False
    elif off > y + epsilon:
        print("Electrolyzer is off but off is 1")
        return False
    elif egrid + R_h2p*h + wind - eelzr + epsilon < demand:
        print("Demand not met")
        return False
    elif eelzr*R_p2h > P2H * y + epsilon:
        print("Electrolyzer exceeds P2H limit")
        return False
    elif h*R_h2p > H2P + epsilon:
        print("H2P exceeds limit")
        return False
    elif h > s + epsilon:
        print("Hydrogen used exceeds storage")
        return False
    elif next_s < -epsilon or next_s > C + epsilon:
        print("Next storage out of bounds")
        return False
    elif eelzr > wind + epsilon:
        print("Electrolyzer exceeds wind supply")
        return False
    return True