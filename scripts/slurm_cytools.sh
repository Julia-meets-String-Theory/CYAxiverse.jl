#!/bin/bash
#SBATCH --job-name=cy_axiverse           # Name of the job
#SBATCH --array=4-40                     # Array range: maps directly to h11 (e.g., 4 to 40)
#SBATCH --time=04:00:00                  # Maximum walltime per array task (HH:MM:SS)
#SBATCH --nodes=1                        # 1 node per h11 task
#SBATCH --ntasks=1                       # 1 task per node
#SBATCH --cpus-per-task=8                # Number of CPU cores per h11 task (matches Python's --cores)
#SBATCH --mem=16G                        # RAM per array task
#SBATCH --output=logs/cy_%A_h11_%a.out   # Standard output log (%A = array job ID, %a = task ID)
#SBATCH --error=logs/cy_%A_h11_%a.err    # Standard error log

# 1. Load your Python environment containing CYTools
# (Modify this line to match your cluster's module system or Conda setup)
# module load python/3.10
# source activate cytools_env

# 2. Define how many geometries to generate per h11
N_POLYTOPES=100

# 3. Define the output directory (use high-performance scratch space if available)
OUTPUT_DIR="/path/to/your/scratch/axiverse_data"

echo "=========================================================="
echo "Starting array task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Generating $N_POLYTOPES geometries for h11=$SLURM_ARRAY_TASK_ID"
echo "=========================================================="

# 4. Execute the Python script
# We pass the SLURM_ARRAY_TASK_ID as both min and max to isolate one h11 per job
python generate_h5.py \
    --h11_min $SLURM_ARRAY_TASK_ID \
    --h11_max $SLURM_ARRAY_TASK_ID \
    --n $N_POLYTOPES \
    --outdir $OUTPUT_DIR \
    --cores $SLURM_CPUS_PER_TASK

echo "Finished array task $SLURM_ARRAY_TASK_ID"