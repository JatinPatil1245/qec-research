# dataset.py
import torch

def generate_2d_data(num_samples=100):
    """
    Generates random (x, y) coordinates between -1 and 1.
    Target = 1 if the point is inside a circle of radius 0.7.
    """
    X = torch.rand((num_samples, 2)) * 2 - 1 
    
    # Calculate radius squared (x^2 + y^2)
    radius_squared = X[:, 0]**2 + X[:, 1]**2
    y = (radius_squared < 0.49).long()
    
    return X, y

if __name__ == "__main__":
    X, y = generate_2d_data(5)
    print("Coordinates (X):\n", X)
    print("Targets (y):\n", y)