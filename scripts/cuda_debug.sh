nvcc -g ./src/cvtree_cuda.cu -o ./bin/cvtree_cuda.bin &&
cuda-gdb -ex=run ./bin/cvtree_cuda.bin
