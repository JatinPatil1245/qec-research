import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

class HybridQuantumEmbedding(nn.Module):
    def __init__(self, classical_dim=128, num_qubits=4):
        super().__init__()
        
        self.fc_in = nn.Linear(classical_dim, num_qubits)
        self.qnn = self._build_quantum_layer(num_qubits)
        self.quantum_layer = TorchConnector(self.qnn)
        self.fc_out = nn.Linear(num_qubits, classical_dim)

    def _build_quantum_layer(self, num_qubits):
        # 1. Use the modern Qiskit 2.x functional API
        feature_map = zz_feature_map(num_qubits, reps=1)
        ansatz = real_amplitudes(num_qubits, reps=1)
        
        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        
        # 2. Define an observable for each individual qubit (Z0, Z1, Z2, Z3)
        observables = [
            SparsePauliOp.from_sparse_list([("Z", [i], 1)], num_qubits=num_qubits) 
            for i in range(num_qubits)
        ]
        
        return EstimatorQNN(
            circuit=qc,
            observables=observables,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )

    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        x = self.quantum_layer(x)
        x = self.fc_out(x)
        return x

if __name__ == "__main__":
    dummy_llm_vector = torch.rand((1, 128))
    print("Classical Input Shape:", dummy_llm_vector.shape)
    
    model = HybridQuantumEmbedding(classical_dim=128, num_qubits=4)
    output = model(dummy_llm_vector)
    
    print("Quantum-Processed Output Shape:", output.shape)
    print("The PyTorch autograd engine is automatically tracking the quantum gates.")