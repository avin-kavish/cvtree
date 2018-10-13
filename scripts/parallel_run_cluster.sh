#!/bin/bash
#PBS -N cvtree_cpu_parallel
#PBS -l ncpus=16,mem=24GB,walltime=01:00
cd $PBS_O_WORKDIR
export MAX_FILE_LOADS=6
g++ -std=c++11 -O3 ./src/improved_parallel.cpp -o ./bin/improved_parallel.bin -pthread -fopenmp -fopt-info &&
./bin/improved_parallel.bin
