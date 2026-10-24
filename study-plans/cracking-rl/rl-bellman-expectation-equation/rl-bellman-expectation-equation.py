def bellman_expectation_backup(P: list, R: list, policy: list, gamma: float, V: list) -> list:
    """
    Returns one updated state value per state, rounded to four decimals.
    """

    V_new = [0.0] * len(V)
    action_len = len(policy[0])

    for s in range(len(V)):
        for action in range(action_len):
            v = 0.0
            
            for sp in range(len(V)):
                v+=P[s][action][sp] * (R[s][action][sp] + gamma * V[sp])
    
            
            V_new[s] += policy[s][action] * v
 
        
    return [round(val, 4) for val in V_new]