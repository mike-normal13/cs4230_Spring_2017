#!/bin/bash
#
#SBATCH --time=0:00:20
#SBATCH --nodes=8
#SBATCH --ntasks=16
#SBATCH --account=soc-kp
#SBATCH --partition=soc-kp
#SBATCH --job-name=some_mpi
#SBATCH --mem=10g
#SBATCH --export=ALL

ulimit -c unlimited -s

mpicc send_recv.c -o send_recv

mpiexec -n 16 ./send_recv