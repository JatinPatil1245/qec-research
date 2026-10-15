from qiskit import QuantumCircuit, transpile

def compile_to_fault_tolerant(circuit: QuantumCircuit, opt_level: int = 2) -> QuantumCircuit:
    """
    Transpiles a logical quantum circuit into a strict Clifford + T basis.
    """
    # The standard fault-tolerant instruction set for surface codes
    ft_basis = ['h', 's', 'cx', 't', 'tdg', 'x', 'z', 'measure']
    
    compiled_circuit = transpile(
        circuit, 
        basis_gates=ft_basis,
        optimization_level=opt_level
    )
    
    return compiled_circuit