def mc_policy_evaluation(episodes: list, gamma: float, num_states: int) -> list:
    """
    Returns first-visit state values rounded to four decimals; unvisited states are 0.0.
    """
    total = [0.0] * num_states
    counts = [0.0] * num_states
    for episode in episodes:
        T = len(episode)

        ## backward pass to compute G_t would be:
        returns_per_step = [0.0] * T
        running = 0
        for t in range(T-1,-1,-1):
            state, reward = episode[t]
            running = (running * gamma) + float(reward)
            returns_per_step[t] =  running

        ## first visit aggregation:
        seen = set()
        for t in range(T):
            s, _ = episode[t]
            if s not in seen:
                seen.add(s)
                total[s] += returns_per_step[t]
                counts[s] += 1

    V = [0.0] * num_states
    for s in range(num_states):
        if counts[s] > 0:
            V[s] = round(total[s]/counts[s],4)

    return V
        