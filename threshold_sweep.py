# threshold_sweep.py
import numpy as np
from stim_simulator import generate_noisy_surface_code
from decoder import batch_decode_errors

def run_sweep():
    distances = [3, 5]
    # Sweep around the expected ~1% threshold (0.2% to 1.6%)
    physical_noise_rates = [0.002, 0.005, 0.008, 0.010, 0.014]
    shots = 5000  # 5,000 shots gives good statistical precision
    
    results = {d: [] for d in distances}
    
    print(f"--- Running QEC Threshold Sweep ({shots} shots per configuration) ---")
    print(f"{'Noise (p)':<12} | " + " | ".join([f"d={d} P_L" for d in distances]))
    print("-" * (14 + 12 * len(distances)))

    for p in physical_noise_rates:
        row = [f"{p:<12.3f}"]
        for d in distances:
            # Code cycles equal to distance (d rounds of measurement)
            circuit = generate_noisy_surface_code(distance=d, rounds=d, noise_rate=p)
            
            # Fast vectorized sampling
            sampler = circuit.compile_detector_sampler()
            detector_samples, observable_samples = sampler.sample(
                shots=shots, 
                separate_observables=True
            )
            
            # Batch MWPM decode
            p_logical = batch_decode_errors(circuit, detector_samples, observable_samples)
            results[d].append(p_logical)
            row.append(f"{p_logical:<10.4f}")
            
        print(" | ".join(row))

    # Optional plotting
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for d in distances:
            plt.plot(physical_noise_rates, results[d], marker='o', label=f'Distance d={d}')
        
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Physical Error Rate (p)')
        plt.ylabel('Logical Error Rate (P_L)')
        plt.title('Surface Code Threshold Sweep (MWPM)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.savefig('threshold_plot.png')
        print("\nPlot saved successfully to 'threshold_plot.png'.")
    except ImportError:
        print("\nMatplotlib not detected. Skipping plot generation.")

if __name__ == "__main__":
    run_sweep()