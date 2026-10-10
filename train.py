# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from micro_qnlp import MicroQuantumTransformer
from dataset import generate_toy_data

def main():
    # Hyperparameters
    vocab_size = 50
    seq_length = 4
    embed_dim = 8
    num_qubits = 4
    batch_size = 32
    epochs = 20

    # Initialize Architecture
    model = MicroQuantumTransformer(
        vocab_size=vocab_size, 
        embed_dim=embed_dim, 
        num_qubits=num_qubits
    )
    
    criterion = nn.CrossEntropyLoss()
    # High learning rate because quantum landscapes can be very flat (barren plateaus)
    optimizer = optim.Adam(model.parameters(), lr=0.05) 

    # Load Data
    X_train, y_train = generate_toy_data(vocab_size, seq_length, num_samples=batch_size * 4)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")
    print("Beginning End-to-End Hybrid Training...\n")

    # Training Loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass
        logits = model(X_train)
        loss = criterion(logits, y_train)
        
        # Backward Pass 
        loss.backward()
        optimizer.step()
        
        # Track accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y_train).float().mean() * 100
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.1f}%")

    print("\nTraining Complete! Saving weights...")
    torch.save(model.state_dict(), "micro_qnlp.pth")
    print("Model saved to 'micro_qnlp.pth'.")

if __name__ == "__main__":
    main()