# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from ft_classifier import FTQuantumClassifier
from dataset import generate_2d_data

def main():
    batch_size = 16
    epochs = 15
    
    # Initialize the 17-qubit architecture
    model = FTQuantumClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1) 

    # Generate the coordinate dataset
    X_train, y_train = generate_2d_data(num_samples=batch_size * 5)

    print("Total Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Beginning 17-Qubit FT-QML Training (Active Mid-Circuit Correction)...\n")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass (Triggers State Prep -> Extraction -> Correction -> Measurement)
        logits = model(X_train)
        loss = criterion(logits, y_train)
        
        # Backward Pass (PyTorch executes the Parameter-Shift rule through the QNode)
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y_train).float().mean() * 100
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.1f}%")

    print("\nTraining Complete! Saving robust logical weights...")
    torch.save(model.state_dict(), "ft_qml_weights.pth")
    print("Model saved to 'ft_qml_weights.pth'.")

if __name__ == "__main__":
    main()