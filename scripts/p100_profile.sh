module load foss
module load cuda/9.0.176
nvcc -std=c++11 -gencode arch=compute_60,code=sm_60 ./src/cvtree_p100.cu -o ./bin/cvtree_p100.bin && 
nvprof ./bin/cvtree_p100.bin 
