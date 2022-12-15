#!/bin/bash
#SBATCH -N 1
#SBATCH -A plgar2022-cpu
#SBATCH --ntasks-per-node=8
#SBATCH -p plgrid
#SBATCH --output=lab6.out

module load scipy-bundle/2021.10-intel-2021b
export SLURM_OVERLAP=1

mpiexec python -m mpi4py.futures lab6.py
