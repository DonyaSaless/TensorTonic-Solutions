def td_zero_update(V: list, transitions: list, alpha: float, gamma: float) -> list:
    """
    Returns the updated state-value list, rounded to four decimals.
    """
 
    for s, r, sp in transitions:
        V[s] = V[s] + alpha * (float(r) + (gamma * V[sp]) - V[s])
    
    return [float(round(v, 4)) for v in V]