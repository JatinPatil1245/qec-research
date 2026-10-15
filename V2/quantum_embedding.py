import torch
import torch.nn as nn
import pennylane as qml
from pennylane import qjit
import numpy as np
import jax.numpy as jnp

print("[INIT] Loading quantum_embedding.py")

# Target the RTX 3060 backend
dev = qml.device("lightning.gpu", wires=17, shots=1000)
print(f"[INIT] PennyLane device initialized: {dev.name}")

# Dummy MWPM cache for the dry run (replace with your actual lookup table generation)
mwpm_cache = np.zeros((256, 9), dtype=int) 

def apply_physical_corrections(corrections):
    pass # Placeholder for your physical corrections

@qjit
@qml.qnode(dev, interface="jax")
def ft_qml_circuit(inputs, weights):
    print("  -> [JIT COMPILE] Tracing: Starting ft_qml_circuit setup")
    
    for i in range(9):
        print(f"  -> [JIT COMPILE] Tracing: Encoding RY/RZ for data qubit {i}")
        qml.RY(inputs[0] * jnp.pi, wires=i)
        qml.RZ(inputs[1] * jnp.pi, wires=i)
        
    for i in range(9):
        print(f"  -> [JIT COMPILE] Tracing: Applying trainable RX weight for wire {i}")
        qml.RX(weights[i], wires=i)
        
    measurements = []
    print("  -> [JIT COMPILE] Tracing: Starting syndrome measurement loop")
    for ancilla_idx in range(9, 17):
        print(f"  -> [JIT COMPILE] Tracing: Entangling and measuring ancilla {ancilla_idx}")
        data_idx = ancilla_idx - 9
        qml.CNOT(wires=[data_idx, ancilla_idx]) 
        qml.CNOT(wires=[data_idx + 1, ancilla_idx])
        m = qml.measure(ancilla_idx)
        measurements.append(m)
        
    print("  -> [JIT COMPILE] Tracing: Calculating syndrome integer")
    syndrome_int = sum(m * (2**j) for j, m in enumerate(measurements))
    
    print("  -> [JIT COMPILE] Tracing: Starting 256-branch MWPM conditional loop")
    for val in range(256):
        print(f"  -> [JIT COMPILE] Tracing: Evaluating MWPM branch {val}/255")
        if sum(mwpm_cache[val]) > 0:
            print(f"  -> [JIT COMPILE] Tracing: Injecting qml.cond graph for branch {val}")
            qml.cond(syndrome_int == val, apply_physical_corrections)(mwpm_cache[val])
            
    print("  -> [JIT COMPILE] Tracing: Finished MWPM loop. Compiling return expectation value")
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

class CatalystQuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        print("    -> [AUTOGRAD FORWARD] Executing Catalyst C++ binary")
        inputs_np = inputs.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        
        out_np = np.array(ft_qml_circuit(inputs_np, weights_np))
        print("    -> [AUTOGRAD FORWARD] Circuit execution complete.")
        
        ctx.save_for_backward(inputs, weights)
        return torch.tensor(out_np, dtype=torch.float32, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output):
        print("    -> [AUTOGRAD BACKWARD] Starting parameter-shift rule")
        inputs, weights = ctx.saved_tensors
        inputs_np = inputs.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        
        shift = np.pi / 2.0
        grad_weights_np = np.zeros_like(weights_np)
        
        for i in range(len(weights_np)):
            print(f"      -> [PARAM SHIFT] Calculating shifts for weight {i}/8")
            w_plus = weights_np.copy()
            w_plus[i] += shift
            out_plus = np.array(ft_qml_circuit(inputs_np, w_plus))
            
            w_minus = weights_np.copy()
            w_minus[i] -= shift
            out_minus = np.array(ft_qml_circuit(inputs_np, w_minus))
            
            grad_weights_np[i] = 0.5 * (out_plus - out_minus)
        
        print("    -> [AUTOGRAD BACKWARD] Parameter-shift complete.")
        grad_weights_tensor = torch.tensor(grad_weights_np, dtype=torch.float32, device=weights.device)
        grad_weights = grad_output * grad_weights_tensor
        
        return None, grad_weights

class FTQuantumEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        print("  -> [INIT] Initializing FTQuantumEmbedding module")
        self.weights = nn.Parameter(torch.randn(9, requires_grad=True))
        
    def forward(self, x):
        print(f"  -> [FORWARD Q-LAYER] Processing batch of size {x.shape[0]}")
        batch_outputs = []
        for i, sample in enumerate(x):
            print(f"    -> [FORWARD Q-LAYER] Sending sample {i} to Catalyst")
            out = CatalystQuantumFunction.apply(sample, self.weights)
            batch_outputs.append(out)
        
        print("  -> [FORWARD Q-LAYER] Batch complete. Stacking outputs.")
        return torch.stack(batch_outputs).unsqueeze(1)