#!/bin/bash
#PBS -N cvtree_cpu_parallel
#PBS -l ncpus=8,mem=24GB,walltime=01:00
cd $PBS_O_WORKDIR
#module load foss
export MAX_FILE_LOADS=6
g++ -std=c++11 ./src/improved_parallel.cpp -o ./bin/improved_parallel.bin -pthread &&
./bin/improved_parallel.bin
