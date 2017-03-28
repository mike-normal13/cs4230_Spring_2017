#!/bin/bash
#
#SBATCH --account=soc-kp
#SBATCH --partition=soc-kp
#SBATCH --job-name=comp_422_openmp   
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10g
#SBATCH --time= 00:10:00
#SBATCH --export=ALL

ulimit -c unlimited -s

mpicc mat_mat_mult.c -o mat_mat_mult

mpiexec -n 2 ./mat_mat_mult