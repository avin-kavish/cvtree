nvcc -std=c++11 -O3 -gencode arch=compute_61,code=sm_61 ./src/cvtree_cuda.cu -o ./bin/cvtree_cuda.bin && 
./bin/cvtree_cuda.bin
