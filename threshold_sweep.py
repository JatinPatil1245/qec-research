# threshold_sweep.py
import argparse
import json
import numpy as np

# Enforce headless rendering before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import stim
from decoder import batch_decode_errors


def run_single_point(distance: int, rounds: int, p: float, shots: int):
    """Generates circuit, samples syndromes, and decodes."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )

    sampler = circuit.compile_detector_sampler()
    detector_samples, observable_samples = sampler.sample(
        shots=shots, separate_observables=True
    )

    p_logical = batch_decode_errors(circuit, detector_samples, observable_samples)
    std_err = np.sqrt((p_logical * (1.0 - p_logical)) / shots)
    return p_logical, std_err


def main():
    parser = argparse.ArgumentParser(description="Surface Code Threshold Sweep")
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=[3, 5],
        help="Code distances to sweep (e.g., --distances 3 5 7)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=10000,
        help="Number of shots per configuration point",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="threshold_sweep",
        help="Prefix for saved plot and data files",
    )
    args = parser.parse_args()

    distances = args.distances
    shots = args.shots
    physical_noise_rates = [0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014]

    results = {str(d): {"p_l": [], "std_err": []} for d in distances}

    print(f"--- Running QEC Threshold Sweep ({shots} shots per point) ---")
    header_cols = [f"{'Noise (p)':<12}"] + [f"d={d} P_L" for d in distances]
    print(" | ".join(header_cols))
    print("-" * (14 + 14 * len(distances)))

    for p in physical_noise_rates:
        row = [f"{p:<12.4f}"]
        for d in distances:
            p_logical, std_err = run_single_point(
                distance=d, rounds=d, p=p, shots=shots
            )
            results[str(d)]["p_l"].append(p_logical)
            results[str(d)]["std_err"].append(std_err)
            row.append(f"{p_logical:.4f}±{std_err:.4f}")
        print(" | ".join(row))

    raw_data = {
        "physical_noise_rates": physical_noise_rates,
        "distances": distances,
        "shots": shots,
        "results": results,
    }
    json_path = f"{args.output_prefix}.json"
    with open(json_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"\nRaw data saved to '{json_path}'.")

    # Generate log-log plot with error bars
    plt.figure(figsize=(8, 6))
    for d in distances:
        p_l_vals = results[str(d)]["p_l"]
        err_vals = results[str(d)]["std_err"]
        plt.errorbar(
            physical_noise_rates,
            p_l_vals,
            yerr=err_vals,
            fmt="-o",
            capsize=3,
            label=f"Distance d={d}",
        )

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Physical Error Rate (p)")
    plt.ylabel("Logical Error Rate (P_L)")
    plt.title("Surface Code Threshold Sweep (MWPM)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    plot_path = f"{args.output_prefix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to '{plot_path}'.")


if __name__ == "__main__":
    main()