#!/bin/bash

# IMPORTANT PARAMS
#SBATCH -p gpu                       # Submit to which partition
#SBATCH --gres=gpu:A100:4            # GPU resources, format TYPE:device:quantity 

# OTHER PARAMS
#SBATCH -J run_NN         # Name the job
#SBATCH -o run_NN-%j.out   # Write the standard output to file named 'jMPItest-<job_number>.out'
#SBATCH -e run_NN-%j.err   # Write the standard error to file named 'jMPItest-<job_number>.err'

# COMMANDS ON THE COMPUTE NODE
pwd                         # prints current working directory
date                        # prints the date and time

# compile the GPU program
srun --container-name=cuda-11.2.2 make
# run the GPU program multiple times
srun --container-name=cuda-11.2.2 ./run_NN.py