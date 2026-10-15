# stim_simulator.py
import stim

def generate_noisy_surface_code(distance: int = 3, rounds: int = 3, noise_rate: float = 0.01) -> stim.Circuit:
    """
    Generates a physical surface code circuit with circuit-level depolarizing noise.
    """
    print(f"\n--- Stim Physical Simulation (d={distance}, rounds={rounds}, noise={noise_rate}) ---")
    
    # Stim has built-in generators for standard QEC topologies.
    # We generate a rotated planar code memory experiment preserving a logical Z state.
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise_rate,
        before_round_data_depolarization=noise_rate,
        before_measure_flip_probability=noise_rate,
        after_reset_flip_probability=noise_rate
    )
    
    print(f"Generated a physical circuit with {len(circuit)} instructions.")
    return circuit

def sample_syndromes(circuit: stim.Circuit, num_shots: int = 5):
    """
    Runs the noisy circuit to extract error syndromes and the final logical observable.
    """
    print(f"Sampling {num_shots} noisy execution shots...")
    
    # A sampler executes the circuit and returns the detector measurements
    sampler = circuit.compile_detector_sampler()
    
    # We sample both the detectors (syndromes) and the logical observable (final answer)
    detector_samples, observable_samples = sampler.sample(shots=num_shots, separate_observables=True)
    
    for i in range(num_shots):
        # Count how many parity checks flagged an error
        defects = sum(detector_samples[i])
        # Check if the final logical qubit flipped (0 = Success, 1 = Logical Error)
        logical_flip = observable_samples[i][0]
        
        print(f"Shot {i+1}: {defects} defects detected | Logical Error: {'YES' if logical_flip else 'NO'}")
        
    return detector_samples, observable_samples