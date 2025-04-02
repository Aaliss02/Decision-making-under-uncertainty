import numpy as np

def feasibility_check(e, on, off, eelzr, h, egrid, s, problemData, wind_trajectory):
    initial_state = problemData.get('initial_electrolyzer_state', 'OFF')
    initial_storage = problemData.get('initial_hydrogen_storage', 0)
    C = problemData['hydrogen_capacity']
    P2H = problemData['p2h_rate']
    R_p2h = problemData['conversion_p2h']
    H2P = problemData['h2p_rate']
    R_h2p = problemData['conversion_h2p']
    T = problemData['num_timeslots']

    feasible = True
    state = initial_state
    current_storage = initial_storage

    for tau in range(1, T + 1):
        current_on = on.get((e, tau), 0)
        current_off = off.get((e, tau), 0)
        current_eelzr = eelzr.get((e, tau), 0)
        current_h = h.get((e, tau), 0)
        current_egrid = egrid.get((e, tau), 0)
        current_s = s.get((e, tau), 0)
        D_t = problemData['demand_schedule'][tau - 1]
        wind_t = wind_trajectory[e][tau - 1]

        # Check on/off are binary and not both 1
        if current_on not in (0, 1) or current_off not in (0, 1):
            print(f"Experiment {e}, time {tau}: Non-binary on/off values")
            feasible = False
        if current_on + current_off > 1:
            print(f"Experiment {e}, time {tau}: Both on and off are 1")
            feasible = False

        # Check electrolyzer state constraints
        if state == 'OFF':
            if current_eelzr > 1e-6:
                print(f"Experiment {e}, time {tau}: Electrolyzer OFF but eelzr={current_eelzr}")
                feasible = False
        else:
            max_eelzr = P2H * R_p2h
            if current_eelzr > max_eelzr + 1e-6:
                print(f"Experiment {e}, time {tau}: eelzr exceeds P2H limit ({max_eelzr})")
                feasible = False

        # Check H2P constraints
        max_h = H2P * R_h2p
        if current_h > max_h + 1e-6:
            print(f"Experiment {e}, time {tau}: h exceeds H2P limit ({max_h})")
            feasible = False
        hydrogen_used = current_h / R_h2p if R_h2p != 0 else 0
        if hydrogen_used > current_storage + 1e-6:
            print(f"Experiment {e}, time {tau}: Hydrogen used {hydrogen_used} > storage {current_storage}")
            feasible = False

        # Check power balance
        power_supply = wind_t + current_egrid + current_h - current_eelzr
        if not np.isclose(power_supply, D_t, atol=1e-3):
            print(f"Experiment {e}, time {tau}: Power imbalance (supply={power_supply}, demand={D_t})")
            feasible = False

        # Check storage consistency and bounds
        if not np.isclose(current_s, current_storage, atol=1e-3):
            print(f"Experiment {e}, time {tau}: Storage mismatch (expected {current_storage}, got {current_s})")
            feasible = False

        hydrogen_produced = current_eelzr * R_p2h
        new_storage = current_storage - hydrogen_used + hydrogen_produced

        if new_storage < -1e-6 or new_storage > C + 1e-6:
            print(f"Experiment {e}, time {tau}: Storage out of bounds ({new_storage})")
            feasible = False

        # Update state and storage for next iteration
        state = 'ON' if current_on else 'OFF' if current_off else state
        current_storage = new_storage

    return feasible