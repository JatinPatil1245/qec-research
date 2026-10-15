from qiskit import QuantumCircuit

def build_toffoli_circuit() -> QuantumCircuit:
    """
    Generates a 3-qubit circuit featuring a non-Clifford Toffoli (CCX) gate.
    """
    qc = QuantumCircuit(3)
    
    # Place the two control qubits into superposition
    qc.h(0)
    qc.h(1)
    
    # Apply the Toffoli gate (Controls: 0, 1 | Target: 2)
    qc.ccx(0, 1, 2)
    
    # Measure all qubits
    qc.measure_all()
    
    return qc