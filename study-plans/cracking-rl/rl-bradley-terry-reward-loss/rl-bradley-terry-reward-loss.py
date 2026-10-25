import math

def bradley_terry_loss(r_chosen: list, r_rejected: list) -> float:
    """
    Returns the mean preference loss as a float rounded to four decimals.
    """
    def lsig(z):
        if z>=0:
            return - math.log1p(math.exp(-z))

        return z - math.log1p(math.exp(z))

    N = len(r_chosen)
    total = 0.0
    for i in range(N):
        total += lsig(r_chosen[i] - r_rejected[i])

    return round(-total/N,4)

    