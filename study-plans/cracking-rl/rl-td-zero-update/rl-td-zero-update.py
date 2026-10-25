def td_zero_update(V: list, transitions: list, alpha: float, gamma: float) -> list:
    """
    Returns the updated state-value list, rounded to four decimals.
    """
 
    V = [float(v) for v in V]
    for s,r,sp in transitions:
            V[s] +=  alpha * (float(r) + (gamma * V[sp]) -V[s])

    return [round(v,4) for v in V]