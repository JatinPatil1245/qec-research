import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ft_classifier import FTQuantumClassifier

print("[TRAIN] Script started.")

def load_qec_data():
    print("[TRAIN] Generating/Loading dataset")
    X = torch.randn(100, 2) 
    y = torch.randint(0, 2, (100,)).float()
    print(f"[TRAIN] Dataset loaded. Features: {X.shape}, Labels: {y.shape}")
    return TensorDataset(X, y)

def main():
    print("[TRAIN] Entering main()...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Hardware assigned: {device}")

    # --- RESTORED FULL DATASET ---
    full_dataset = load_qec_data()
    print(f"[TRAIN] Using full dataset with {len(full_dataset)} samples.")

    dataloader = DataLoader(
        full_dataset, 
        batch_size=4,       
        shuffle=True, 
        num_workers=0,      
        drop_last=True
    )
    print(f"[TRAIN] DataLoader initialized with {len(dataloader)} batches.")

    print("[TRAIN] Moving model to device...")
    model = FTQuantumClassifier().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    print("[TRAIN] Optimizer and Loss Function ready.")

    epochs = 5
    for epoch in range(epochs):
        print(f"\n[EPOCH {epoch+1}] Starting epoch")
        model.train()
        
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            print(f"\n  [BATCH {batch_idx+1}] Moving data to device")
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).long()
            
            print(f"  [BATCH {batch_idx+1}] Zeroing gradients")
            optimizer.zero_grad()
            
            print(f"  [BATCH {batch_idx+1}] Triggering Forward Pass")
            outputs = model(X_batch).squeeze()
            
            print(f"  [BATCH {batch_idx+1}] Computing Loss")
            loss = criterion(outputs, y_batch)
            print(f"  [BATCH {batch_idx+1}] Loss computed: {loss.item()}")
            
            print(f"  [BATCH {batch_idx+1}] Triggering Backward Pass (Autograd)")
            loss.backward()
            
            print(f"  [BATCH {batch_idx+1}] Stepping Optimizer")
            optimizer.step()
            
            print(f"  [BATCH {batch_idx+1}] Batch complete")
            
        print(f"[EPOCH {epoch+1}] Epoch complete.\n")

if __name__ == "__main__":
    main()