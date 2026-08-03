# How to use this:

Create a logs directory: SLURM will silently fail if the folder for the output files doesn't exist. Before submitting, run:
```bash

mkdir logs
```
Update the environment: Make sure you uncomment and modify the `module load` or `source activate` lines so the compute node knows where `cytools` lives. Compute nodes rarely share the login node's default Python environment.

Submit the job:
```bash

sbatch submit_axiverse.sh
```