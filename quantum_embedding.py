# quantum_embedding.py
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import pymatching
from tqdm import tqdm

# 17 qubits: 0-8 Data, 9-16 Ancilla
dev = qml.device("default.qubit", wires=17)

# 1. Define the Parity Check Matrix for a d=3 surface code
# (8 syndrome measurements for 9 data qubits)
# 1. Define the Parity Check Matrix for the 1D Repetition Circuit
# (Matches the CNOT ladder in ft_qml_circuit perfectly)
H = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1],
], dtype=np.uint8)

matching = pymatching.Matching.from_check_matrix(H)

# 2. Generate the 256-item MWPM Lookup Table
print("Generating MWPM Lookup Table...")
mwpm_cache = []
for i in range(256):
    # Convert integer [0-255] into an 8-bit array
    syndrome = np.array([(i >> j) & 1 for j in range(8)], dtype=np.uint8)
    
    # Run the MWPM algorithm to get the 9-bit correction array
    correction = matching.decode(syndrome)
    mwpm_cache.append(correction.tolist())
print("Lookup Table Generated Successfully.\n")

# 3. Define the physical correction applicator
def apply_physical_corrections(correction_bits):
    for data_idx in range(9):
        if correction_bits[data_idx] == 1:
            qml.PauliX(wires=data_idx)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def ft_qml_circuit(inputs, weights):
    # Data Encoding
    for i in range(9):
        qml.RY(inputs[0] * np.pi, wires=i)
        qml.RZ(inputs[1] * np.pi, wires=i)
    
    # Trainable AI Layer
    for i in range(9):
        qml.RX(weights[i], wires=i)
    
    # Mid-Circuit Syndrome Extraction
    measurements = []
    for ancilla_idx in range(9, 17):
        data_idx = ancilla_idx - 9
        qml.CNOT(wires=[data_idx, ancilla_idx]) 
        qml.CNOT(wires=[data_idx + 1, ancilla_idx])
        
        m = qml.measure(ancilla_idx)
        measurements.append(m)
        
    # Calculate the integer index of the syndrome [0-255]
    syndrome_int = sum(m * (2**j) for j, m in enumerate(measurements))

    # Apply the pre-calculated MWPM correction via PennyLane's conditional tape
    for val in range(256):
        qml.cond(syndrome_int == val, apply_physical_corrections)(mwpm_cache[val])
        
    # Logical Measurement
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

class FTQuantumEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(9, requires_grad=True))
        
    def forward(self, x):
        batch_outputs = []
        for sample in tqdm(x, desc="Evaluating Quantum Circuits", leave=False):
            out = ft_qml_circuit(sample, self.weights)
            batch_outputs.append(out)
        
        return torch.stack(batch_outputs).unsqueeze(1)