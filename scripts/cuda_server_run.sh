nvcc -gencode arch=compute_37,code=sm_60 ./src/cvtree_cuda_server.cu -o ./bin/cvtree_cuda_server.bin && 
./bin/cvtree_cuda_server.bin
