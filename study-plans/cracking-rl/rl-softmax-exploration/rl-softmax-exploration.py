import math

def softmax_action_probs(Q_values: list, tau: float) -> list:
    """
    Returns the action-probability list, rounded to four decimals.
    """
    max_Q = max(Q_values)
    exps = [math.exp((q-max_Q)/tau) for q in Q_values]
    s = sum(exps)
    return [round(e/s,4) for e in exps]

    