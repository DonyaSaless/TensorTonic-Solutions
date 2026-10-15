import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    """
    Returns a DataLoader with inverse-frequency sampling and replacement.
    """
    class_counts = torch.bincount(labels)
    class_weight = 1.0 / class_counts.float()
    sample_weight = class_weight[labels]
    sampler = WeightedRandomSampler(sample_weight, num_samples = len(labels), replacement = True)
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset,batch_size =batch_size, sampler= sampler)
    
