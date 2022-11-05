#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -p plgrid

module load scipy-bundle/2021.10-intel-2021b
export SLURM_OVERLAP=1

n_array=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 )
n_size=( 64 128 256 512 )
for n in "${n_array[@]}"
do
	for size in "${n_size[@]}"
	do
		mpiexec -n $n python lab4_solver.py $size 100
	done
done
