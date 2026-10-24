def bellman_optimality_backup(P: list, R: list, gamma: float, V: list) -> list:
    """
    Returns one updated state value per state, rounded to four decimals.
    """
    len_states = len(V)
    len_actions = len(P[0])

    V_new = [0.0] * len_states
    
    for s in range(len_states):
        best_action = float("-inf")
        for action in range(len_actions):
            v_action = 0.0
            for sp in range(len_states):
                v_action += P[s][action][sp] * (R[s][action][sp] + gamma * V[sp])
            if v_action > best_action:
                best_action = v_action

        
        V_new[s] =  round(best_action,4)

    return V_new