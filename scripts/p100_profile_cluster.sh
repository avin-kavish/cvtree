#!/bin/bash -l
#PBS -N cvtree_profiling
#PBS -l ncpus=1
#PBS -l mem=4gb
#PBS -l walltime=00:40
#PBS -l ngpus=1
#PBS -l gputype=P100
cd $PBS_O_WORKDIR
module load foss
module load cuda/9.1.85
nvcc -std=c++11 -gencode arch=compute_60,code=sm_60 ./src/cvtree_p100.cu -o ./bin/cvtree_p100.bin && 
nvprof ./bin/cvtree_p100.bin 
