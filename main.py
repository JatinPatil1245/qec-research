from circuit_builder import build_toffoli_circuit
from compiler import compile_to_fault_tolerant

def main():
    # Step 1: Generate the logical circuit
    print("--- Logical Circuit ---")
    logical_qc = build_toffoli_circuit()
    print(logical_qc.draw())
    print(f"Logical Gate Count: {logical_qc.count_ops()}")
    print(f"Logical Depth: {logical_qc.depth()}\n")

    # Step 2: Push through the fault-tolerant compiler
    print("--- Fault-Tolerant Compiled Circuit ---")
    compiled_qc = compile_to_fault_tolerant(logical_qc)
    
    # The drawn circuit will be large, so we print the analytics first
    print(f"Compiled Gate Count: {compiled_qc.count_ops()}")
    print(f"Compiled Depth: {compiled_qc.depth()}\n")
    
    # Draw the massive decomposed circuit
    print(compiled_qc.draw(fold=-1)) # fold=-1 prevents text wrapping for easier reading

if __name__ == "__main__":
    main()