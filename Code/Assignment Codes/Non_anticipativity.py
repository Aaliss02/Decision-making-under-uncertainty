def Non_anticipativity(nb_scen, L, wind, price):
    S = {}

    for s in range(nb_scen):
        for t in range(1, L+1):
            if (s, t) not in S:
                S[s, t] = set()
                
            for s_prime in range(nb_scen):
                belongs = True
                for u in range(1, t+1):
                    if wind[s][t-u] != wind[s_prime][t-u] or price[s][t-u] != price[s_prime][t-u]:
                        belongs = False
                        break 
                if belongs: 
                    S[s, t].add(s_prime) 
    return S  
