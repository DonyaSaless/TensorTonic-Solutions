import torch

def dense_layer_forward(inputs: torch.Tensor, weight_matrix: torch.Tensor, biases: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    """
    Returns an output vector in neuron order, preserving input dtype and device.
    """
    a = torch.matmul(inputs, weight_matrix.T) + biases
    a = torch.tanh(a) if nonlinear else a
    return a

    
    
