nvcc -gencode arch=compute_37,code=sm_60 ./src/cvtree_cuda.cu -o ./bin/cvtree_cuda.bin && 
./bin/cvtree_cuda.bin
