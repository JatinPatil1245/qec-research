# decoder.py
import pymatching
import numpy as np

def batch_decode_errors(stim_circuit, detector_samples, observable_samples) -> float:
    """
    Decodes all shots concurrently using PyMatching's vectorized C++ backend.
    Returns the logical error rate.
    """
    matching = pymatching.Matching.from_stim_circuit(stim_circuit)
    
    # Vectorized decode across all shots simultaneously
    predicted_observables = matching.decode_batch(detector_samples)
    
    # Calculate how many shots failed
    # observable_samples shape is (shots, num_observables)
    num_errors = np.sum(np.any(predicted_observables != observable_samples, axis=1))
    
    return float(num_errors / len(detector_samples))