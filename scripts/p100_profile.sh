nvcc -gencode arch=compute_61,code=sm_61 ./src/cvtree_p100.cu -o ./bin/cvtree_p100.bin && 
nvprof ./bin/cvtree_p100.bin 
