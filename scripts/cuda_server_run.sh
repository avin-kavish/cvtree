nvcc -gencode arch=compute_37,code=sm_37 ./src/cvtree_cuda_server.cu -o ./bin/cvtree_cuda_server.bin && 
./bin/cvtree_cuda_server.bin
