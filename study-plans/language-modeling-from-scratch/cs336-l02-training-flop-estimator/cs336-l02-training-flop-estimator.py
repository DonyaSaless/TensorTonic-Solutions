def flop_estimator(matmuls: list[list[int]], attention_flops: int = 0) -> dict:
    """
    Returns integer forward_flops, backward_flops, and total_flops in a dictionary.
    """
    frw_flops = attention_flops
    for entry in matmuls:
        (B,D,K) = entry
        frw_flops = frw_flops + (2*B*D*K)

    backward_flops = frw_flops * 2
    total = backward_flops + frw_flops

    return {'forward_flops': frw_flops, 'backward_flops': backward_flops, 'total_flops': total}
        
