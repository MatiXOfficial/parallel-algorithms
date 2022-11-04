#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -p plgrid

module load scipy-bundle/2021.10-intel-2021b

n_array=( 1 4 16 )
n_size=( 128 256 )
for n in "${n_array[@]}"
do
	for size in "${n_size[@]}"
	do
		mpiexec -n $n python script.py $size 100
	done
done

