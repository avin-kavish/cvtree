nvcc -gencode arch=compute_50,code=sm_50 ./src/cvtree_cuda.cu -o ./bin/cvtree_cuda.bin && 
./bin/cvtree_cuda.bin
