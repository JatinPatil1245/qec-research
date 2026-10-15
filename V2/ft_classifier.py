import torch
import torch.nn as nn
from quantum_embedding import FTQuantumEmbedding

print("[INIT] Loading ft_classifier.py...")

class FTQuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        print("[INIT] Building FTQuantumClassifier architecture...")
        
        print("  -> [INIT] Setting up pre-processing Linear layer...")
        self.pre_net = nn.Linear(2, 2) # Example: Maps raw data to RY/RZ angles
        
        self.q_layer = FTQuantumEmbedding()
        
        print("  -> [INIT] Setting up post-processing Linear layer...")
        self.post_net = nn.Linear(1, 2) # Maps single expectation value to 2 class logits
        
    def forward(self, x):
        print("[FORWARD CLASSIFIER] Input received. Shape:", x.shape)
        
        x = self.pre_net(x)
        print("[FORWARD CLASSIFIER] Pre-net complete. Shape:", x.shape)
        
        x = self.q_layer(x)
        print("[FORWARD CLASSIFIER] Quantum layer complete. Shape:", x.shape)
        
        x = self.post_net(x)
        print("[FORWARD CLASSIFIER] Post-net complete. Final output shape:", x.shape)
        
        return x