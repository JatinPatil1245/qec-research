import torch
from torch.utils.data import Dataset, DataLoader

class CircleDataset(Dataset):
    def __init__(self, num_samples=400):
        # Generate 2D coordinates between -1.0 and 1.0
        self.X = torch.rand((num_samples, 2)) * 2 - 1
        # Label 1 if inside radius 0.7 (r^2 < 0.49), else 0
        self.y = (self.X[:, 0]**2 + self.X[:, 1]**2 < 0.49).long()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloader(batch_size=40, num_workers=0, pin_memory=False):
    dataset = CircleDataset()
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )