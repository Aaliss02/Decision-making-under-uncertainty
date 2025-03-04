import numpy as np

def Non_anticipativity(Scenarios, L, wind, price):
    S = {}

    for s in Scenarios:
        for t in range(1, L+1):
            for s_prime in Scenarios:
                belongs = True
                for u in range(1, t+1):
                    if wind[s][t-u] != wind[s_prime][t-u] or price[s][t-u] != price[s_prime][t-u]:
                        belongs = False
                if belongs: 
                    S[s, t].add(s_prime)
    return S  
