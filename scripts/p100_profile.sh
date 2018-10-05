nvcc -std=c++11 -gencode arch=compute_60,code=sm_60 ./src/cvtree_p100.cu -o ./bin/cvtree_p100.bin && 
nvprof ./bin/cvtree_p100.bin 
