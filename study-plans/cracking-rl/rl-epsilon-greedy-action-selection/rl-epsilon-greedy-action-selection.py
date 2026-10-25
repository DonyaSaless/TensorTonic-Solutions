def epsilon_greedy_probs(Q_values: list, epsilon: float) -> list:
    """
    Returns the action-probability list, rounded to four decimals.
    """
    a_star =  Q_values.index(max(Q_values))
    A = len(Q_values)
    probs = [epsilon/A] * A
    probs[a_star] += (1.0-epsilon)

    return [round(prob,4) for prob in probs]