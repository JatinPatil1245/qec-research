# main.py
from circuit_builder import build_toffoli_circuit
from compiler import compile_to_fault_tolerant
from lattice_mapper import map_to_surface_code
from stim_simulator import generate_noisy_surface_code, sample_syndromes
from decoder import decode_errors

def main():
    # Step 1: Generate the logical circuit
    print("--- Logical Circuit ---")
    logical_qc = build_toffoli_circuit()
    print(logical_qc.draw())
    print(f"Logical Gate Count: {logical_qc.count_ops()}")
    
    # Step 2: Push through the fault-tolerant compiler
    print("\n--- Fault-Tolerant Compiled Circuit ---")
    compiled_qc = compile_to_fault_tolerant(logical_qc)
    print(compiled_qc.draw(fold=-1))
    print(f"Compiled Gate Count: {compiled_qc.count_ops()}")
    
    # Step 3: Map to the physical hardware topology
    routing_plan = map_to_surface_code(compiled_qc, distance=3)
    
    # Step 4: Generate physical stabilizer circuits and inject noise
    # (Running at 0.2% noise to test below-threshold performance)
    stim_circuit = generate_noisy_surface_code(distance=3, rounds=3, noise_rate=0.002)
    detectors, observables = sample_syndromes(stim_circuit, num_shots=100)
    
    # Step 5: Feed the physical defects into the PyMatching software decoder
    decode_errors(stim_circuit, detectors, observables)

if __name__ == "__main__":
    main()