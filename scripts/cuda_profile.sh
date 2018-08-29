nvcc -gencode arch=compute_61,code=sm_61 ./src/cvtree_cuda.cu -o ./bin/cvtree_cuda.bin && 
nvprof ./bin/cvtree_cuda.bin
