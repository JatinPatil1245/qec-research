#!/bin/bash
#SBATCH --job-name=qec_threshold
#SBATCH --nodes=1
#SBATCH --ntasks=16                 # Request multi-threaded processing
#SBATCH --mem=64G                   # High memory allocation for PyMatching graphs
#SBATCH --time=24:00:00             # Max wall-clock time
#SBATCH --output=sweep_results_%j.out # Standard output and error log

echo "Starting QEC Threshold Sweep on BluePebble..."
echo "Date: $(date)"

# Activate your existing virtual environment
source ~/qec-env/bin/activate

# Execute the heavy Python sweep
python threshold_sweep.py --distances 3 5 7 --shots 50000 --output-prefix bluepebble_run

echo "Sweep completed."
echo "Date: $(date)"