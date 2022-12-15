#!/bin/bash
#SBATCH -N 1
#SBATCH -A plgar2022-cpu
#SBATCH --ntasks-per-node=30
#SBATCH -p plgrid-testing
#SBATCH --output=lab6.out

rm results.csv

module load scipy-bundle/2021.10-intel-2021b
export SLURM_OVERLAP=1

d_list=( 2 3 4 )
n_proc=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 )
for d in "${d_list[@]}"
do
	for n in "${n_proc[@]}"
	do
		mpiexec -n $n python -m mpi4py.futures lab6.py $n 10 $d results.csv
	done
done
