# dataset.py
import torch

def generate_toy_data(vocab_size=50, seq_length=4, num_samples=100):
    """
    Generates synthetic sequences. 
    The logic rule: If the sum of the tokens in the sequence is greater 
    than half the max possible sum, the target class is 1. Otherwise, 0.
    """
    # Generate random token sequences
    X = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    # Apply the logical rule to create targets
    threshold = (vocab_size * seq_length) / 2
    y = (X.sum(dim=1) > threshold).long()
    
    return X, y

if __name__ == "__main__":
    X, y = generate_toy_data(num_samples=5)
    print("Sample Sequences (X):\n", X)
    print("Sample Targets (y):\n", y)