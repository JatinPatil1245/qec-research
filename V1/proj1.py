from qiskit import QuantumCircuit, transpile

# 1. Define the hidden bitstring (we'll use '11' for 2 input qubits)
secret_string = "11"
num_inputs = len(secret_string)

# Create circuit: 2 input qubits + 1 ancilla qubit, and 2 classical bits
qc = QuantumCircuit(num_inputs + 1, num_inputs)

# 2. State Preparation
# Inputs start at |0>, ancilla needs to be flipped to |1>
qc.x(num_inputs)

# Apply Hadamard to all qubits to create superposition
qc.h(range(num_inputs + 1))
qc.barrier()

# 3. The Oracle
# Qiskit uses little-endian ordering (qubit 0 is the rightmost bit)
for i, bit in enumerate(reversed(secret_string)):
    if bit == '1':
        # Apply CX from the input qubit to the ancilla
        qc.cx(i, num_inputs)
qc.barrier()

# 4. Interference and Measurement
# Apply final Hadamard to inputs and measure
qc.h(range(num_inputs))
qc.measure(range(num_inputs), range(num_inputs))

print("--- Logical Circuit ---")
print(qc.draw())

# 5. Fault-Tolerant Compilation
# Restrict the transpiler strictly to a Clifford + T basis
ft_basis = ['h', 's', 'cx', 't', 'tdg', 'x', 'z', 'measure']

# optimization_level=2 collapses adjacent gates to reduce depth
compiled_qc = transpile(qc, basis_gates=ft_basis, optimization_level=2)

print("\n--- Fault-Tolerant Compiled Circuit ---")
print(compiled_qc.draw())
print("\nGate Counts:", compiled_qc.count_ops())
print("Circuit Depth:", compiled_qc.depth())