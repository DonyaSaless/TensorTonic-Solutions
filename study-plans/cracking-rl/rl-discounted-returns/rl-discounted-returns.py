def discounted_returns(rewards: list, gamma: float) -> list:
    """
    Returns one discounted return per reward, rounded to four decimals.
    """
    T = len(rewards)
    G = [0.0] * T
    running = 0.0
    for t in range(T-1,-1,-1):
        running = running * gamma + float(rewards[t])
        G[t] = round(running, 4)
    return G
    