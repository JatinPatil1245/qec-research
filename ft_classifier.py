# ft_classifier.py
import torch
import torch.nn as nn
from quantum_embedding import FTQuantumEmbedding 

class FTQuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum_core = FTQuantumEmbedding()
        
        # Projects the single logical qubit expectation value into 2 class logits
        self.classifier = nn.Linear(1, 2)

    def forward(self, x):
        # x shape: (batch_size, 2)
        # Cast the PennyLane Double tensor to a PyTorch Float tensor
        logical_expectation = self.quantum_core(x).float() 
        
        # Output shape: (batch_size, 2)
        return self.classifier(logical_expectation)